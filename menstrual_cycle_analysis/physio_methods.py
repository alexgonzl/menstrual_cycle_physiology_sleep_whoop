"""`PhysioMethods` — slice needed by notebook 04 (VAR S9).

Source: `whoop_analyses/whoop_analyses/physio_methods.py` (PhysioMethods).

This port covers what notebook 04 uses end-to-end:
  - constructor that augments a CycleBehavMethods instance with biometric
    setup (`CORE_BIOMETRICS`, `_filters`, `plotting_physio_labels_short`,
    `PLOT_DPI`, `palettes`, etc.).
  - `_define_filter` (Butterworth/FIR factory) + `filter_data`.
  - `_init_plotting_labels`.
  - `get_reference_table` + `_get_user_x_cycle_bounds` + `add_column_reference_table`.
  - `process_physio_data` (interpolate + filter biometrics, add `m_*`,
    `pct_*`, `b_*`, `z_*`, `rz_*` columns).
  - `_zscore`.

GAM machinery and Fig 3 plot helpers are intentionally not included; they
will be ported when notebook 03 is implemented.
"""
from __future__ import annotations

import itertools
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from tqdm import tqdm

from . import config
from ._plot_utils import (
    add_legend,
    draw_rectangle_gradient,
    fixed_yticks,
    setup_axes,
    single_var_point_plot,
)
from .cl_behav_methods import CycleBehavMethods
from .stats.circular import circmean_day, circvar_day


class PhysioMethods(CycleBehavMethods):
    """Minimal port; carries biometric setup and filter factories used by
    `Biometrics_VAR` for S9.
    """

    CORE_BIOMETRICS = ['RHR', 'HRV', 'RR', 'skin_temp', 'blood_oxygen']
    FILTER_PRESETS = {
        'biometric':  dict(ftype='iir', iir_order=1, causal=False, linear_detrend=False, w0=1/90, w1=1/7),
        'var':        dict(ftype='iir', iir_order=2, causal=True,  linear_detrend=False, w0=1/90, w1=1/3),
        'behavioral': dict(ftype='fir', fir_length=7, causal=True,  linear_detrend=True,  w1=1/7),
    }
    interp_win = 7
    buffer_win = 35

    def __init__(self, cbm: CycleBehavMethods | None = None, *args, **kwargs):
        if cbm is None:
            super().__init__(*args, **kwargs)
        else:
            self.__dict__.update(cbm.__dict__)

        self.CL_bin_edges = np.arange(21, 35.5, 2)
        self.CL_bins = [22, 24, 26, 28, 30, 32, 34]
        self.base_range = np.arange(-self.buffer_win, self.buffer_win + 1)
        self.reference_table = None
        self.m_BIOMETRICS = [f'm_{vv}' for vv in self.CORE_BIOMETRICS]
        self._filters = {}
        for name, params in self.FILTER_PRESETS.items():
            self._define_filter(**params)
            self._filters[name] = self.filter_data
        self.filter_data = self._filters['biometric']

        self.delta_cl_palette = sns.diverging_palette(250, 30, l=60, s=90, center="dark", n=7)

        self.locking_params = dict(follicular=dict(d_range=np.arange(-15, 35),
                                                    delta_cl_col='delta_cl_binned_f',
                                                    xticks=np.arange(-10, 31, 10)),
                                   luteal=dict(d_range=np.arange(-30, 16),
                                               delta_cl_col='delta_cl_binned_b',
                                               xticks=np.arange(-30, 11, 10)))

        self.centering_col = self._c_col
        self._user_x_cycle_bounds = None

        self.sleep_vars = ['sleep_dur', 'sl_onset']
        self.wo_vars = ['total_intensity', 'total_wo_duration', 'n_wo']

        age_mpl_cmap = plt.get_cmap('viridis')
        age_mpl_colors = [age_mpl_cmap(i) for i in np.linspace(0.05, 0.95, 8)]
        self.palettes = dict(
            cl=sns.diverging_palette(250, 30, l=60, s=90, center="dark", n=len(self.CL_bins)),
            age=age_mpl_colors,
        )
        self._init_plotting_labels()

        self.gam_models = None
        self.gam_test_data = None
        self.gam_ages = np.arange(18, 51)
        self.gam_CLs = np.arange(21, 36)
        self.gam_vars = ['d', 'cl', 'age']
        self.gam_min_max_table = None
        self.gam_additional_covariates = []
        self.gam_data = None
        self.gam_models_metrics = None

        self._c_window = 21
        self._c2_window = self._c_window * 3
        self._c2_window_min = int(self._c2_window * 2 / 3)
        self._a_window = self._c_window // 3

    def _init_plotting_labels(self):
        base_labels = {
            'RHR': 'Resting HR',
            'skin_temp': 'Temp.',
            'RR': 'Respiration Rate',
            'HRV': 'HR Variability',
            'blood_oxygen': 'Blood Oxygen',
            'age': 'Age',
            'cl': 'Cycle Length',
        }

        units = {
            'RHR': 'beats/min',
            'skin_temp': 'C',
            'RR': 'breaths/min',
            'HRV': 'ms',
            'blood_oxygen': '%',
            'cl': 'days',
            'z': 'z'
        }

        base_labels2 = {
            'RHR': 'RHR',
            'skin_temp': 'Temp.',
            'RR': 'RR',
            'HRV': 'HRV',
            'blood_oxygen': r'Blood $O^2$',
        }

        self.plotting_physio_labels = base_labels.copy()
        self.plotting_physio_labels_units = {}
        self.plotting_physio_labels_short_units = {}

        prefixes = ['m_', 'z_', 'b_', 'rz_', 'pct_']
        for prefix in prefixes:
            for key, label in base_labels.items():
                if key in ['age', 'cl']:
                    continue
                prefixed_key = f"{prefix}{key}"
                self.plotting_physio_labels[prefixed_key] = label

                if prefix == 'z_':
                    unit = units['z']
                elif prefix == 'pct_':
                    unit = "%"
                else:
                    unit = units[key]
                self.plotting_physio_labels_units[prefixed_key] = f"{label} [{unit}]"
                self.plotting_physio_labels_short_units[prefixed_key] = f"{base_labels2[key]} [{unit}]"

        for key, label in base_labels.items():
            if key in units:
                self.plotting_physio_labels_units[key] = f"{label} [{units[key]}]"

        for key, label in base_labels2.items():
            if key in units:
                self.plotting_physio_labels_short_units[key] = f"{label} [{units[key]}]"

        self.plotting_physio_labels_short = base_labels2

    def _define_filter(self, ftype='iir', w0=None, w1=None, causal=True,
                       iir_order=2, fir_length=7, linear_detrend=False):
        if w1 is None:
            w1 = 1 / self.interp_win
        lowpass_only = w0 is None

        if ftype == 'iir':
            if lowpass_only:
                sos = signal.butter(iir_order, w1, btype='low', fs=1, output='sos')
            else:
                sos = signal.butter(iir_order, [w0, w1], btype='band', fs=1, output='sos')
            zi = signal.sosfilt_zi(sos)

            def filter_fn(y1):
                if linear_detrend:
                    y1 = signal.detrend(y1, type='linear')
                if causal:
                    y2, _ = signal.sosfilt(sos, y1, zi=zi * y1[0])
                else:
                    y2 = signal.sosfiltfilt(sos, y1)
                return y2

        elif ftype == 'fir':
            if lowpass_only:
                taps = signal.firwin(fir_length, w1, window='hann', fs=1)
            else:
                taps = signal.firwin(fir_length, [w0, w1], pass_zero=False, fs=1)
            delay = (fir_length - 1) // 2
            zi = signal.lfilter_zi(taps, 1)

            def filter_fn(y1):
                if linear_detrend:
                    y1 = signal.detrend(y1, type='linear')
                if causal:
                    y2, _ = signal.lfilter(taps, 1.0, y1, zi=zi * y1[0])
                    y2 = np.roll(y2, -delay)
                    y2[-delay:] = np.nan
                else:
                    y2 = signal.filtfilt(taps, 1.0, y1)
                return y2

        else:
            raise ValueError(f"Unknown ftype: {ftype!r}. Use 'iir' or 'fir'.")

        def filter_data(x: pd.Series):
            m = np.nanmean(x)
            y = x.interpolate(limit=self.interp_win)
            y1 = y.fillna(m).values
            y2 = filter_fn(y1)
            y3 = pd.Series(y2)
            y3[y.isna().values] = np.nan
            return y3.values

        self.filter_data = filter_data

    def _zscore(self, x):
        if len(x) > 3:
            mean = np.nanmean(x)
            sd = np.nanstd(x)
            if sd == 0:
                sd = 1
            return (x - mean) / sd
        else:
            return np.nan

    # ------------------------------------------------------------------
    # Reference table + biometric processing pipeline (cells 8-9 of source nb)
    # ------------------------------------------------------------------
    def _get_user_x_cycle_bounds(self):
        if self._user_x_cycle_bounds is not None:
            return self._user_x_cycle_bounds

        user_cycle_group_idx = self.tables['cycle'].index
        tg = self.data.groupby("n_id")
        base_range = self.base_range

        bounds_data = {}

        print("Getting cycle bounds (optimized)")
        for user in tqdm(self.users):
            d = tg.get_group(user)
            user_data_index = d.index.values
            d = d.reset_index()
            data_range_max = len(d) - 1

            c_locs = d.index[d[self.centering_col] == 1].to_numpy()
            c_num = d.loc[d[self.centering_col] == 1, self._cn_col].to_numpy()
            nC = len(c_locs)

            if nC == 0:
                continue

            if nC == 1:
                c_bounds_start = max(c_locs[0] - self.buffer_win, 0)
                c_bounds_end = data_range_max + 1
                c_bounds_list = [np.arange(c_bounds_start, c_bounds_end)]
            else:
                c_bounds_list = []

                c_bounds_start = max(c_locs[0] - self.buffer_win, 0)
                c_bounds_end = c_locs[1]
                c_bounds_list.append(np.arange(c_bounds_start, c_bounds_end))

                for ii in range(1, nC - 1):
                    c_bounds_start = c_locs[ii - 1]
                    c_bounds_end = c_locs[ii + 1]
                    c_bounds_list.append(np.arange(c_bounds_start, c_bounds_end))

                c_bounds_start = c_locs[nC - 2]
                c_bounds_end = min(c_locs[nC - 1] + self.buffer_win, data_range_max + 1)
                c_bounds_list.append(np.arange(c_bounds_start, c_bounds_end))

            for ii, (cn, cc, c_bounds) in enumerate(zip(c_num, c_locs, c_bounds_list)):
                cycle_range = cc + base_range

                range_idx = np.isin(cycle_range, c_bounds)
                valid_cycle_positions = cycle_range[range_idx]

                valid_mask = (valid_cycle_positions >= 0) & (valid_cycle_positions <= data_range_max)
                valid_positions = valid_cycle_positions[valid_mask]

                if len(valid_positions) == 0:
                    continue

                idx = (user, cn)
                if idx in user_cycle_group_idx:
                    bounds_data[idx] = {}
                    valid_base_positions = base_range[range_idx][valid_mask]
                    valid_data_indices = user_data_index[valid_positions]

                    bounds_data[idx].update(dict(zip(valid_base_positions, valid_data_indices)))

        print("Converting to DataFrame...")
        if bounds_data:
            out = pd.DataFrame.from_dict(bounds_data, orient='index')
            out = out.reindex(index=user_cycle_group_idx, columns=base_range)
        else:
            out = pd.DataFrame(index=user_cycle_group_idx, columns=base_range, dtype=object)

        self._user_x_cycle_bounds = out
        return out

    def get_reference_table(self, overwrite=False):
        "reference table that contains a buffer of days x cycle x subject"
        if not overwrite:
            if self.reference_table is not None:
                return self.reference_table

        user_cycle_bounds = self._get_user_x_cycle_bounds()
        ref_cols = np.setdiff1d(self.tables['cycle'].columns, self.delta_cl_thr_cols + self.z_delta_cl_thr_cols).tolist()
        table = self.tables['cycle'][ref_cols]
        user_cycle_bounds = user_cycle_bounds.loc[table.index]
        table = pd.concat((user_cycle_bounds, table), axis=1)
        table = table.reset_index()
        out = table.melt(id_vars=['n_id', self._cn_col] + ref_cols, var_name='cycle_day', value_name='data_index')
        out.dropna(subset='data_index', inplace=True)

        out['cl_bin'] = pd.cut(out['length'], self.CL_bin_edges, labels=self.CL_bins)
        self.reference_table = out

        return out

    def add_column_reference_table(self, col):
        self.reference_table[col] = self.reference_table.data_index.map(self.data[col])

    def add_sleep_2_ref(self):
        self.add_column_reference_table('sl_onset')
        self.add_column_reference_table('sleep_dur')

    def add_wo_2_ref(self):
        self.add_column_reference_table('eTRIMP')
        self.add_column_reference_table('norm_eTRIMP')
        self.add_column_reference_table('total_wo_duration')
        self.add_column_reference_table('n_morning_wo')
        self.add_column_reference_table('n_wo')

    def add_wo_data(self):
        eTRIMP_HR_WEIGHTS = pd.Series(index=config.HR_ZONES, data=[0, 1, 2, 3, 4, 5])
        self.data['eTRIMP'] = (self.data[config.HR_ZONES] * eTRIMP_HR_WEIGHTS).sum(axis=1)
        self.data['norm_eTRIMP'] = self.data['eTRIMP'] / self.data['total_wo_duration']
        self.data['norm_eTRIMP'] = self.data['norm_eTRIMP'].fillna(0)
        self.data['wo_dur'] = self.data['total_wo_duration']

    def add_seasonalilty_2_ref(self):
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['day_of_year'] = self.data['date'].dt.dayofyear
        self.data['cos_season'] = np.cos(2 * np.pi * self.data['day_of_year'] / 365.25)
        self.data['sin_season'] = np.sin(2 * np.pi * self.data['day_of_year'] / 365.25)
        self.data['week_day'] = self.data['date'].dt.weekday
        self.data['weekend'] = self.data['week_day'].isin([5, 6]).astype(int)

        for col in ['day_of_year', 'cos_season', 'sin_season', 'week_day', 'weekend']:
            self.add_column_reference_table(col)

    def add_acute_chronic_behaviors_data(self):
        for vv in ['eTRIMP', 'norm_eTRIMP', 'wo_dur', 'sleep_dur']:
            self.data[f'c_{vv}'] = self.data.groupby("n_id")[vv].rolling(self._c_window, min_periods=self._c_window).mean().reset_index(0, drop=True)
            self.data[f'c_{vv}_p1'] = self.data[f'c_{vv}'].shift(1)
            self.data[f'c2_{vv}_p1'] = self.data.groupby("n_id")[vv].rolling(self._c2_window, min_periods=self._c2_window_min).mean().reset_index(0, drop=True).shift(1)
            self.data[f'a_{vv}'] = self.data.groupby("n_id")[vv].rolling(self._a_window, min_periods=self._a_window).mean().reset_index(0, drop=True)
            self.data[f'acr_{vv}'] = (self.data[f'a_{vv}'] - self.data[f'c_{vv}_p1']) / self.data[f'c_{vv}_p1'] * 100

    def add_acute_chronic_behaviors_ref(self):
        for vv in ['eTRIMP', 'norm_eTRIMP', 'wo_dur', 'sleep_dur']:
            self.add_column_reference_table(f'c_{vv}')
            self.add_column_reference_table(f'c_{vv}_p1')
            self.add_column_reference_table(f'c2_{vv}_p1')
            self.add_column_reference_table(f'a_{vv}')
            self.add_column_reference_table(f'acr_{vv}')

    def add_menstrual_phases_ref(self, window_days=7):
        self.reference_table['phase'] = 'other'

        idx = (self.reference_table.cycle_day >= -window_days) & (self.reference_table.cycle_day <= -1)
        self.reference_table.loc[idx, 'phase'] = 'premenstrual'

        idx = (self.reference_table.cycle_day >= 0) & (self.reference_table.cycle_day < window_days)
        self.reference_table.loc[idx, 'phase'] = 'menstrual'

        idx = (self.reference_table.cycle_day >= window_days) & (self.reference_table.cycle_day < 2 * window_days)
        self.reference_table.loc[idx, 'phase'] = 'postmenstrual'

    def _add_phase_segment(self, ax, phase, phase_days=None,
                           phase_color=None, alpha=0.5):
        if phase_days is None:
            phase_days = self.phase_days[phase]

        if phase_color is None:
            color = self.phase_colors[phase]
        else:
            color = phase_color
        ax.axvspan(phase_days[0], phase_days[1] + 1, facecolor=color, alpha=alpha, lw=0, zorder=-1)

    def process_physio_data(self, preset='biometric', prefix=None,
                            types=('m', 'pct', 'z', 'rz', 'b'),
                            overwrite=False, **filter_kwargs):
        prev_filter = self.filter_data
        if preset is not None:
            self.filter_data = self._filters[preset]
        elif filter_kwargs:
            self._define_filter(**filter_kwargs)

        def _col(typ, biometric):
            if prefix:
                return f'{typ}_{prefix}_{biometric}'
            return f'{typ}_{biometric}'

        t = self.data
        tg = t.groupby("n_id")

        for biometric in self.CORE_BIOMETRICS:
            cols = [_col(typ, biometric) for typ in types]
            if (not overwrite) and all(c in t.columns for c in cols):
                continue

            print(f"Processing {biometric}" + (f" [{prefix}]" if prefix else ""))
            for user in tqdm(self.users):
                idx = tg.get_group(user).index
                d = t.loc[idx]
                raw = d[biometric]
                filtered = self.filter_data(raw)
                baseline = raw.mean()

                if 'm' in types:
                    t.loc[idx, _col('m', biometric)] = filtered
                if 'b' in types:
                    t.loc[idx, _col('b', biometric)] = baseline + filtered
                if 'pct' in types:
                    t.loc[idx, _col('pct', biometric)] = filtered / baseline * 100
                if 'z' in types:
                    t.loc[idx, _col('z', biometric)] = self._zscore(filtered)

            if 'rz' in types and 'z' in types:
                sd_biometric = tg[biometric].std().mean()
                t[_col('rz', biometric)] = sd_biometric * t[_col('z', biometric)]

            for typ in types:
                self.add_column_reference_table(_col(typ, biometric))

        self.filter_data = prev_filter

    # GAM methods
    def gam_cycle_model_bam_full(self, gam_data, y_var, ar_lags=None,
                                 additional_covariates=None,
                                 use_parallel=False, n_cores=2, rho=0.3,
                                 verbose=False):
        """
        Fit BAM model with correlation structure - simplified for scientific work
        """
        try:
            from rpy2.robjects.packages import importr
            import rpy2.robjects as ro
            from rpy2.robjects import default_converter, pandas2ri, numpy2ri
            from rpy2.robjects.conversion import localconverter


            # pandas2ri.activate()
            # numpy2ri.activate()

            with localconverter(default_converter + pandas2ri.converter + numpy2ri.converter):
                mgcv = importr('mgcv')
                base = importr('base')
                stats = importr('stats')
        except Exception as e:
            raise ImportError(f"R packages not available: {e}")

        # Handle inputs
        if ar_lags is None or ar_lags == 0:
            ar_lags = []
        elif isinstance(ar_lags, int) and ar_lags > 0:
            ar_lags = list(range(1, ar_lags + 1))

        if additional_covariates is None:
            additional_covariates = []
        elif isinstance(additional_covariates, str):
            additional_covariates = [additional_covariates]

        # Prepare data - critical for correlation structure
        gam_data_work = gam_data.copy()

        # Create a unique identifier for each subject-cycle combination
        gam_data_work['subject_cycle'] = gam_data_work['n_id'].astype(str) + '_' + gam_data_work['cl'].astype(str)

        # Sort by subject-cycle and then by day - ESSENTIAL for AR correlation
        gam_data_work = gam_data_work.sort_values(['subject_cycle', 'd']).reset_index(drop=True)

        # Create lag variables if needed
        for lag in ar_lags:
            lag_col = f'{y_var}_lag{lag}'
            gam_data_work[lag_col] = gam_data_work.groupby(['subject_cycle'])[y_var].shift(lag)

        lag_cols = [f'{y_var}_lag{lag}' for lag in ar_lags]

        # Define columns and clean data
        required_cols = ['d', 'cl', 'age', y_var, 'subject_cycle']
        all_cols = required_cols + lag_cols + additional_covariates

        # Remove missing values and re-sort
        clean_subset = [y_var] + lag_cols + additional_covariates
        gam_data_clean = gam_data_work.dropna(subset=clean_subset)
        gam_data_clean = gam_data_clean.sort_values(['subject_cycle', 'd']).reset_index(drop=True)

        # Create AR_start logical vector - TRUE at first observation of each subject-cycle
        # This marks where each independent AR1 section begins
        gam_data_clean['AR_start'] = ~gam_data_clean['subject_cycle'].duplicated()

        # Convert logical to R logical vector
        ar_start_logical = ro.BoolVector(gam_data_clean['AR_start'].values)

        if verbose:
            print(f"Fitting BAM on {len(gam_data_clean)} observations")
            print(f"From {gam_data_clean['n_id'].nunique()} subjects")
            print(f"Across {gam_data_clean['subject_cycle'].nunique()} subject-cycle combinations")
            print(f"AR sections: {gam_data_clean['AR_start'].sum()}")

        model_metrics = {
        'n_obs': len(gam_data_clean),
        'n_subjects': gam_data_clean['n_id'].nunique(),
        'n_cycles': gam_data_clean['subject_cycle'].nunique(),
        'ar_sections': gam_data_clean['AR_start'].sum()
        }

        # Convert to R
        with localconverter(pandas2ri.converter):
            r_data = pandas2ri.py2rpy(gam_data_clean[all_cols])

        # Build formula
        formula_parts = [
            "s(d, k=7, bs='cr')",
            "s(cl, k=4, bs='cr')",
            "s(age, k=4, bs='cr')",
            "ti(d, cl, k=c(6,4), bs=c('cr','cr'))",
            "ti(d, age, k=c(6,4), bs=c('cr','cr'))",
            "ti(d, cl, age, k=c(6,4,4), bs=c('cr','cr','cr'))"
        ]

        # Add lag terms if present
        for lag_col in lag_cols:
            formula_parts.append(f"s({lag_col}, k=4, bs='cr')")
            formula_parts.append(f"ti(d, {lag_col}, k=c(6,4), bs=c('cr','cr'))")

        # Add extra covariates
        for covar in additional_covariates:
            formula_parts.append(f"s({covar}, k=4, bs='cr')")
            formula_parts.append(f"ti(d, {covar}, k=c(6,4), bs=c('cr','cr'))")

        # Specific interactions
        if 'sleep_dur' in additional_covariates and 'eTRIMP' in additional_covariates:
            formula_parts.append("ti(sleep_dur, eTRIMP, k=c(4,4), bs=c('cr','cr'))")
            formula_parts.append("ti(d, sleep_dur, eTRIMP, k=c(6,4,4), bs=c('cr','cr','cr'))")
            formula_parts.append("ti(d, sleep_dur, eTRIMP, cl, k=c(6,4,4,4), bs=c('cr','cr','cr', 'cr'))")

        if 'sleep_dur' in additional_covariates and 'c_sleep_dur' in additional_covariates:
            formula_parts.append("ti(sleep_dur, c_sleep_dur, k=c(4,4), bs=c('cr','cr'))")
            formula_parts.append("ti(d, sleep_dur, c_sleep_dur, k=c(6,4,4), bs=c('cr','cr','cr'))")
            formula_parts.append("ti(d, sleep_dur, c_sleep_dur, cl, k=c(6,4,4,4), bs=c('cr','cr','cr', 'cr'))")

        if 'eTRIMP' in additional_covariates and 'c_eTRIMP' in additional_covariates:
            formula_parts.append("ti(eTRIMP, c_eTRIMP, k=c(4,4), bs=c('cr','cr'))")
            formula_parts.append("ti(d, eTRIMP, c_eTRIMP, k=c(6,4,4), bs=c('cr','cr','cr'))")
            formula_parts.append("ti(d, eTRIMP, c_eTRIMP, cl, k=c(6,4,4,4), bs=c('cr','cr','cr', 'cr'))")

        formula_str = f"{y_var} ~ " + " + ".join(formula_parts)

        with localconverter(default_converter):
            formula = ro.r(formula_str)

        if verbose:
            print(f"Formula: {formula_str}")

        # Fit with correlation structure
        if verbose:
            print("Fitting with AR(1) correlation structure...")

        if use_parallel:
            bam_model = mgcv.bam(
                formula,
                data=r_data,
                method="fREML",
                discrete=True,
                nthreads=n_cores,
                AR_start=ar_start_logical,
                rho=rho,
                select=True,
            )
        else:
            bam_model = mgcv.bam(
                formula,
                data=r_data,
                method="fREML",
                discrete=True,
                AR_start=ar_start_logical,
                rho=rho,
                select=True
            )

        if verbose:
            print("BAM model fitting completed successfully")

        # Basic stats
        try:
            edf_sum = base.sum(bam_model.rx2('edf'))[0]
            aic_val = stats.AIC(bam_model)[0]
            dev_expl = base.summary(bam_model).rx2('dev.expl')[0] * 100
            if verbose:
                print(f"Effective degrees of freedom: {edf_sum:.2f}")
                print(f"AIC: {aic_val:.2f}")
                print(f"Deviance explained: {dev_expl:.1f}%")
            model_metrics['edf'] = edf_sum
            model_metrics['D2'] = dev_expl
        except:
            print("Model fitted but couldn't extract all statistics")

        return bam_model, model_metrics

    def predict_bam_chunked_ci(self, bam_model, new_data, chunk_size=100000, alpha=0.05, se_fit=True, unconditional=True):
        """
        Make predictions using BAM model with optional confidence intervals
        """

        try:
            from rpy2.robjects.packages import importr
            import rpy2.robjects as ro
            from rpy2.robjects import default_converter, pandas2ri, numpy2ri
            from rpy2.robjects.conversion import localconverter


            # pandas2ri.activate()
            # numpy2ri.activate()

            with localconverter(default_converter + pandas2ri.converter + numpy2ri.converter):
                mgcv = importr('mgcv')
                stats = importr('stats')

        except Exception as e:
            raise ImportError(f"R packages not available: {e}")

        total_rows = len(new_data)

        all_predictions = []
        all_se = []
        print(f"Making predictions with CI for {total_rows} observations")

        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_data = new_data.iloc[start_idx:end_idx]

            with localconverter(pandas2ri.converter):
                r_chunk = pandas2ri.py2rpy(chunk_data)

            with localconverter(default_converter):
                pred_result = mgcv.predict_bam(bam_model, newdata=r_chunk,
                                            se_fit=se_fit, unconditional=unconditional)

            if se_fit:

                with localconverter(default_converter + numpy2ri.converter + pandas2ri.converter):
                    pred_result_np = pred_result.rx2('fit')
                    pred_result_se_fit_np = pred_result.rx2('se.fit')

                all_predictions.extend(np.array(pred_result_np))
                all_se.extend(np.array(pred_result_se_fit_np))
            else:
                all_predictions = pred_result
                all_se = np.nan * np.zeros_like(all_predictions)

            if (start_idx // chunk_size + 1) % 10 == 0:
                print(f"Processed {end_idx}/{total_rows} predictions")

        predictions = np.array(all_predictions)
        standard_errors = np.array(all_se)

        # Get appropriate critical value
        try:
            with localconverter(default_converter):
                df_residual = bam_model.rx2('df.residual')[0]
                t_value = stats.qt(1 - alpha/2, df_residual)[0]
        except:
            t_value = stats.qnorm(1 - alpha/2)[0]

        margin_error = t_value * standard_errors

        return pd.DataFrame({
            'predictions': predictions,
            'se': standard_errors,
            'lower': predictions - margin_error,
            'upper': predictions + margin_error
        })

    def summary_bam(self, bam_model):
        """
        Get comprehensive model summary from BAM
        """
        from rpy2.robjects.packages import importr
        from rpy2.robjects import default_converter, pandas2ri, numpy2ri
        from rpy2.robjects.conversion import localconverter

        with localconverter(default_converter + pandas2ri.converter + numpy2ri.converter):
            base = importr('base')
            mgcv = importr('mgcv')
            stats = importr('stats')

        print("BAM Model Summary:")
        print("==================")

        try:
            # Print summary
            summary_obj = base.summary(bam_model)
            print(summary_obj)

            # Extract key statistics
            try:
                aic = stats.AIC(bam_model)[0]
            except:
                aic = bam_model.rx2('aic')[0]

            edf_vals = bam_model.rx2('edf')
            edf_sum = base.sum(edf_vals)[0]
            deviance_explained = summary_obj.rx2('dev.expl')[0] * 100

            print(f"\nKey Statistics:")
            print(f"AIC: {aic:.2f}")
            print(f"Total EDF: {edf_sum:.2f}")
            print(f"Deviance explained: {deviance_explained:.1f}%")

            # Get individual term EDFs if available
            try:
                term_names = list(summary_obj.rx2('s.table').rownames)
                print(f"\nSmooth Term EDFs:")
                for i, (name, edf) in enumerate(zip(term_names, edf_vals)):
                    if i < len(term_names):
                        print(f"  {name}: {edf:.2f}")
            except Exception as e:
                print(f"Could not extract term names: {e}")

        except Exception as e:
            print(f"Error getting detailed summary: {e}")
            try:
                print(base.summary(bam_model))
            except Exception as e2:
                print(f"Could not generate any summary: {e2}")
                try:
                    print(f"Model fitted successfully")
                    print(f"Formula: {bam_model.rx2('formula')}")
                except:
                    print("BAM model object exists but summary unavailable")

    def prep_data_gam_cycle_model(self, prefix, additional_covariates=None):
        """ Prepares data for GAM cycle model.
        """
        # if max_lag is None:
        #     max_lag = 3
        # elif max_lag < 0:
        #     raise ValueError("max_lag must be a non-negative integer")
        _min_day = -10
        _max_day = 35
        min_day = _min_day# - max_lag
        max_day = _max_day

        t = self.reference_table.copy(deep=True)
        data_cols = [f"{prefix}_{vv}" for vv in self.CORE_BIOMETRICS]
        if additional_covariates is not None:
            if isinstance(additional_covariates, str):
                additional_covariates = [additional_covariates]
            data_cols += additional_covariates

        if 'eTRIMP' in data_cols: # shift eTRIMP to previous day to align with physiological data
            t['eTRIMP'] = t.groupby("n_id")['eTRIMP'].shift(1)


        gam_data = t[(t.cycle_day <= max_day) & (t.cycle_day >= min_day) & (t.length >= self.MIN_MEDIAN_CL) & (t.length <= self.MAX_MEDIAN_CL)].groupby(["n_id", "length", 'cycle_day'])[data_cols].mean().reset_index()

        gam_data['age'] = gam_data.n_id.map(self.tables['user'].age)

        gam_data.rename(columns={'length': 'cl', 'cycle_day': 'd'}, inplace=True)

        self.gam_data = gam_data
        return gam_data

    def fit_gam_models(self, prefix='pct', additional_covariates=None, overwrite=False):
        """ Fits GAM models for core biometrics.
        """

        biometrics = [f"{prefix}_{vv}" for vv in self.CORE_BIOMETRICS]
        if self.gam_models is not None:
            if not overwrite and all(b in self.gam_models for b in biometrics):
                print("GAM models already fitted")
                return
        else:
            self.gam_models = {}
            self.gam_models_metrics = pd.DataFrame(columns=['n_obs', 'n_subjects', 'n_subject_cycles', 'D2', 'edf'])
            self.gam_additional_covariates = additional_covariates if additional_covariates is not None else []

        gam_data= self.prep_data_gam_cycle_model(prefix=prefix, additional_covariates=additional_covariates)

        if gam_data.empty:
            print("No data available for GAM model")
            return

        for y_var in biometrics:

            print("\n--------------------------------------------------\n")
            print(f"Fitting GAM model for {y_var}...")
            self.gam_models[y_var], self.gam_models_metrics.loc[y_var] = self.gam_cycle_model_bam_full(gam_data, y_var=y_var, additional_covariates=additional_covariates, use_parallel=False, rho=0.3)

            d2 = self.gam_models_metrics.loc[y_var, 'D2']
            edf = self.gam_models_metrics.loc[y_var, 'edf']
            print(f"Model for {y_var} fitted successfully.")
            print(f"Model metrics for {y_var}: D2={d2:0.2f}, edf={edf:0.2f}")
            print("\n--------------------------------------------------\n")

    def get_gam_predictions(self, data, se_fit=False, biometrics=None):
        """ Generates predictions from fitted GAM models for core biometrics.
        data: DataFrame with columns: 'cl', 'age', and 'd'.
        se_fit: If True, returns standard errors and confidence intervals.
        Returns a DataFrame with predictions for each core biometric.
        The DataFrame will have columns named like 'pct_HRV_pred', 'pct_HRV_se', 'pct_HRV_lower', 'pct_HRV_upper' if se_fit is True.
        """
        if self.gam_models is None:
            print("Run fit_gam_models first")
            return

        if se_fit:
            pred_cols = ['pred', 'se', 'lower', 'upper']
        else:
            pred_cols = ['pred']

        if biometrics is None:
            biometrics = self.gam_models.keys()
        elif isinstance(biometrics, str):
            biometrics = [biometrics]

        for y_var in biometrics:
            print(y_var)
            model = self.gam_models[y_var]
            pred_df = self.predict_bam_chunked_ci(model, data, unconditional=True, se_fit=se_fit)

            _pred_cols = [f"{y_var}_{col}" for col in pred_cols]
            if se_fit:
                data[_pred_cols] = pred_df[['predictions', 'se', 'lower', 'upper']].values
            else:
                data[_pred_cols] = pred_df['predictions'].values.reshape(-1, 1)

        return data

    def get_gam_cl_age_sim_data(self, ages=None, cycle_lengths=None, fixed_vals_dict=None):
        """ Generates simulated data for GAM models based on cycle lengths and ages.
        This function creates a DataFrame with cycle lengths and ages, and optionally fills in fixed values
        for specific columns. It then uses the GAM models to generate predictions for these combinations.
        The resulting DataFrame contains the cycle lengths, ages, and the predicted values for each core biometric.
        If fixed values are provided, they will be added to the DataFrame before predictions.
        """
        if ages is None:
            ages = self.gam_ages
        if cycle_lengths is None:
            cycle_lengths = self.gam_CLs
        sim_data = self._gen_gam_sim_data(ages=ages, cycle_lengths=cycle_lengths, fixed_vals_dict=fixed_vals_dict)
        sim_data = self.get_gam_predictions(sim_data, se_fit=False)

        self.gam_sim_data_results = sim_data
        return sim_data

    def get_gam_sim_data_for_vars(self, var_grid_dict, fixed_vals_dict=None, se_fit=False, biometrics=None):
        """
        Generates simulated data for GAM models for any combination of variables.

        Parameters
        ----------
        var_grid_dict : dict
            Dictionary where keys are variable names (e.g., 'age', 'cl', 'd') and values are arrays/lists of values to simulate.
            Example: {'age': np.arange(18, 51), 'cl': np.arange(21, 36), 'd': np.arange(-10, 36)}
        fixed_vals_dict : dict, optional
            Dictionary of fixed values for other variables (e.g., {'sleep_dur': 7.5}).
        se_fit : bool, optional
            If True, returns standard errors and confidence intervals.

        Returns
        -------
        pd.DataFrame
            DataFrame with all combinations of variables and predictions for each core biometric.
        """

        # Ensure 'd' (day) is always present and default to -10 to 35 if not provided
        var_grid_dict = dict(var_grid_dict)  # make a copy to avoid mutating input
        if 'd' not in var_grid_dict:
            var_grid_dict['d'] = np.arange(-10, 36)

        keys = list(var_grid_dict.keys())
        values = [np.asarray(var_grid_dict[k]) for k in keys]
        combos = list(itertools.product(*values))
        sim_data = pd.DataFrame(combos, columns=keys)
        #return sim_data

        # Add fixed values if provided
        if fixed_vals_dict is not None:
            for k, v in fixed_vals_dict.items():
                sim_data[k] = v
        # Fill in any missing columns required by the model with their mean if available
        if hasattr(self, "gam_data"):
            for col in getattr(self, "gam_additional_covariates", []):
                if col not in sim_data.columns:
                    sim_data[col] = self.gam_data[col].mean()

        # filter out invalid days outside of cycle
        sim_data = sim_data[sim_data['d'] <= sim_data['cl']]

        #return sim_data
        sim_data = self.get_gam_predictions(sim_data, se_fit=se_fit, biometrics=biometrics)
        #self.gam_sim_data_results = sim_data
        return sim_data

    def _gen_gam_sim_data(self, ages=None, cycle_lengths=None, day_step=0.5, fixed_vals_dict=None):

        if ages is None:
            ages = np.arange(20, 50, 0.25)
        elif not hasattr(ages, '__iter__'):
            ages = [ages]

        if cycle_lengths is None:
            cycle_lengths = np.arange(self.MIN_MEDIAN_CL, self.MAX_MEDIAN_CL, 0.25)
        elif not hasattr(cycle_lengths, '_'):
            cycle_lengths = [cycle_lengths]

        cycle_days = np.arange(-10, self.MAX_MEDIAN_CL, day_step)

        # Create a meshprocess_physio_datagrid of all combinations
        cd, cl, a = np.meshgrid(cycle_days, cycle_lengths, ages, indexing='ij')

        # Flatten the arrays and create a DataFrame
        sim_data = pd.DataFrame({
            'd': cd.ravel(),
            'cl': cl.ravel(),
            'age': a.ravel()
        })

        for kk in self.gam_additional_covariates:
            sim_data[kk] = self.gam_data[kk].mean()

        if fixed_vals_dict is not None:
            for key, value in fixed_vals_dict.items():
                sim_data[key] = value

        # throw out cycle days > cycle length
        sim_data = sim_data[sim_data.d < sim_data.cl].reset_index(drop=True)
        return sim_data

    def compute_gam_range_contrast_r(self, biometric,
                                     range_conditions, search_ranges=None,
                                     fixed_covariates=None):
        """
        Compute contrast between physiological ranges across different conditions,
        including SE and CI for min and max contrasts.
        """

        gam_model = self.gam_models[biometric]

        import rpy2.robjects as ro
        from rpy2.robjects import default_converter, pandas2ri
        from rpy2.robjects.conversion import localconverter

        if search_ranges is None:
            search_ranges = {
                'min_days': list(range(0, 30)),
                'max_days': list(range(0, 30))
            }

        # Fill in any missing columns required by the model with their mean if available
        if fixed_covariates is None:
            fixed_covariates = {}

        for col in getattr(self, "gam_additional_covariates", []):
            if col not in fixed_covariates.keys():
                fixed_covariates[col] = self.gam_data[col].mean()

        if 'age' in range_conditions[0] or 'age' in range_conditions[1]:
            ages = [cond.get('age') for cond in range_conditions]
            cycle_lengths = [fixed_covariates.get('cl', 28)]

        elif 'cl' in range_conditions[0] or 'cl' in range_conditions[1]:
            cycle_lengths = [cond.get('cl') for cond in range_conditions]
            ages = [fixed_covariates.get('age', 32)]
        else:
            raise ValueError("Must vary either 'age' or 'cl' in range_conditions")

        all_days = list(set(search_ranges['min_days'] + search_ranges['max_days']))
        pred_data = self.get_gam_sim_data_for_vars(var_grid_dict={'d': all_days,
            'age': ages, 'cl': cycle_lengths}, fixed_vals_dict=fixed_covariates, biometrics=biometric)

        pred_data['condition_id'] = 0
        if len(ages) > 1:
            pred_data.loc[pred_data['age'] == ages[1], 'condition_id'] = 1
        elif len(cycle_lengths) > 1:
            pred_data.loc[pred_data['cl'] == cycle_lengths[1], 'condition_id'] = 1
        with localconverter(default_converter + pandas2ri.converter):
            r_pred_data = pandas2ri.py2rpy(pred_data)
        ro.r.assign('pred_data', r_pred_data)
        ro.r.assign('gam_model', gam_model)
        ro.r.assign('min_days', ro.IntVector(search_ranges['min_days']))
        ro.r.assign('max_days', ro.IntVector(search_ranges['max_days']))
        ro.r.assign('n_conditions', len(range_conditions))
        r_code = """
        library(mgcv)
        preds <- predict(gam_model, newdata=pred_data, se.fit=TRUE)
        pred_data$pred <- preds$fit
        X <- predict(gam_model, newdata=pred_data, type="lpmatrix")
        ranges <- numeric(n_conditions)
        min_rows <- numeric(n_conditions)
        max_rows <- numeric(n_conditions)
        min_days_found <- numeric(n_conditions)
        max_days_found <- numeric(n_conditions)
        min_vals <- numeric(n_conditions)
        max_vals <- numeric(n_conditions)
        for(i in 0:(n_conditions-1)) {
            condition_data <- pred_data[pred_data$condition_id == i, ]
            min_subset <- condition_data[condition_data$d %in% min_days, ]
            min_idx <- which.min(min_subset$pred)
            min_rows[i+1] <- which(pred_data$condition_id == i &
                                pred_data$d == min_subset$d[min_idx])
            min_days_found[i+1] <- min_subset$d[min_idx]
            min_vals[i+1] <- min_subset$pred[min_idx]
            max_subset <- condition_data[condition_data$d %in% max_days, ]
            max_idx <- which.max(max_subset$pred)
            max_rows[i+1] <- which(pred_data$condition_id == i &
                                pred_data$d == max_subset$d[max_idx])
            max_days_found[i+1] <- max_subset$d[max_idx]
            max_vals[i+1] <- max_subset$pred[max_idx]
            ranges[i+1] <- max_subset$pred[max_idx] - min_subset$pred[min_idx]
        }
        Vb <- vcov(gam_model)
        # Contrasts
        min_contrast_vector <- X[min_rows[n_conditions],] - X[min_rows[1],]
        max_contrast_vector <- X[max_rows[n_conditions],] - X[max_rows[1],]
        range_contrast_vector <- (X[max_rows[n_conditions],] - X[min_rows[n_conditions],]) -
                                (X[max_rows[1],] - X[min_rows[1],])
        min_contrast <- min_vals[n_conditions] - min_vals[1]
        max_contrast <- max_vals[n_conditions] - max_vals[1]
        range_contrast <- ranges[n_conditions] - ranges[1]
        min_var <- t(min_contrast_vector) %*% Vb %*% min_contrast_vector
        max_var <- t(max_contrast_vector) %*% Vb %*% max_contrast_vector
        range_var <- t(range_contrast_vector) %*% Vb %*% range_contrast_vector
        sigma_resid <- sqrt(gam_model$sig2)
        list(
            range_contrast = range_contrast,
            se = as.numeric(sqrt(range_var)),
            min_contrast = min_contrast,
            min_se = as.numeric(sqrt(min_var)),
            max_contrast = max_contrast,
            max_se = as.numeric(sqrt(max_var)),
            sigma_resid = sigma_resid,
            ranges = ranges,
            min_rows = min_rows,
            max_rows = max_rows,
            min_days_found = min_days_found,
            max_days_found = max_days_found,
            min_vals = min_vals,
            max_vals = max_vals
        )
        """
        try:
            result = ro.r(r_code)
            df_resid = gam_model.rx2('df.residual')[0]
            t_crit = ro.r('qt')(0.975, df_resid)[0]
            sigma_resid = result.rx2('sigma_resid')[0]

            range_c = result.rx2('range_contrast')[0]
            range_se = result.rx2('se')[0]
            min_c = result.rx2('min_contrast')[0]
            min_se = result.rx2('min_se')[0]
            max_c = result.rx2('max_contrast')[0]
            max_se = result.rx2('max_se')[0]

            return {
                'range_contrast': range_c,
                'se': range_se,
                'ci_lower': range_c - t_crit * range_se,
                'ci_upper': range_c + t_crit * range_se,
                'cohens_d': range_c / sigma_resid,
                'min_contrast': min_c,
                'min_se': min_se,
                'min_ci_lower': min_c - t_crit * min_se,
                'min_ci_upper': min_c + t_crit * min_se,
                'max_contrast': max_c,
                'max_se': max_se,
                'max_ci_lower': max_c - t_crit * max_se,
                'max_ci_upper': max_c + t_crit * max_se,
                'sigma_resid': sigma_resid,
                'individual_ranges': np.array(result.rx2('ranges')),
                'conditions': range_conditions,
                'min_days_found': np.array(result.rx2('min_days_found')),
                'max_days_found': np.array(result.rx2('max_days_found')),
                'min_vals': np.array(result.rx2('min_vals')),
                'max_vals': np.array(result.rx2('max_vals')),
            }
        except Exception as e:
            print(f"R execution failed: {e}")
            print("\nPython prediction data we're passing to R:")
            print(pred_data.dtypes)
            print(pred_data.describe())
            try:
                ro.r.assign('test_pred_data', r_pred_data)
                test_result = ro.r('predict(gam_model, newdata=test_pred_data)')
                print("Basic prediction succeeded, data format should be OK")
            except Exception as e2:
                print(f"Basic prediction failed: {e2}")
            raise e

    def print_gam_age_contrast(self, biometric, age1=24, age2=44, cl=28):
        age_range_contrast = self.compute_gam_range_contrast_r(
            biometric=biometric,
            range_conditions=[
                {'age': age1},
                {'age': age2}
            ],
            fixed_covariates={'cl': cl}
        )
        print("-----")
        print(f"Age={age1}")
        print(f"range_val={age_range_contrast['individual_ranges'][0]:.3f}")
        print(f"min_day={age_range_contrast['min_days_found'][0]:.3f}")
        print(f"max_day={age_range_contrast['max_days_found'][0]:.3f}")
        print(f"min_val={age_range_contrast['min_vals'][0]:.3f}")
        print(f"max_val={age_range_contrast['max_vals'][0]:.3f}")

        print("-----")
        print(f"Age={age2}")
        print(f"range_val={age_range_contrast['individual_ranges'][1]:.3f}")
        print(f"min_day={age_range_contrast['min_days_found'][1]:.3f}")
        print(f"max_day={age_range_contrast['max_days_found'][1]:.3f}")
        print(f"min_val={age_range_contrast['min_vals'][1]:.3f}")
        print(f"max_val={age_range_contrast['max_vals'][1]:.3f}")

        print("-----")
        print(f"Comparing ages {age1} and {age2} at CL={cl}")
        print(f"Range effect on range: {age_range_contrast['range_contrast']:.3f} ± {age_range_contrast['se']:.3f}")
        print(f"Confidence Interval: {age_range_contrast['ci_lower']:.3f} - {age_range_contrast['ci_upper']:.3f}")
        print(f"cohens_d={age_range_contrast['cohens_d']:.3f}")

    def print_gam_cl_contrast(self, biometric, cl1=24, cl2=34, age=32):
        cl_range_contrast = self.compute_gam_range_contrast_r(
            biometric=biometric,
            range_conditions=[
                {'cl': cl1},
                {'cl': cl2}
            ],
            fixed_covariates={'age': age}
        )

        print("-----")
        print(f"CL={cl1}")
        print(f"range_val={cl_range_contrast['individual_ranges'][0]:.3f}")
        print(f"min_day={cl_range_contrast['min_days_found'][0]:.3f}")
        print(f"max_day={cl_range_contrast['max_days_found'][0]:.3f}")
        print(f"min_val={cl_range_contrast['min_vals'][0]:.3f}")
        print(f"max_val={cl_range_contrast['max_vals'][0]:.3f}")

        print("-----")
        print(f"CL={cl2}")
        print(f"range_val={cl_range_contrast['individual_ranges'][1]:.3f}")
        print(f"min_day={cl_range_contrast['min_days_found'][1]:.3f}")
        print(f"max_day={cl_range_contrast['max_days_found'][1]:.3f}")
        print(f"min_val={cl_range_contrast['min_vals'][1]:.3f}")
        print(f"max_val={cl_range_contrast['max_vals'][1]:.3f}")

        print("-----")
        print(f"Comparing CL={cl1} and CL={cl2} at Age={age}")
        print(f"Range effect on range: {cl_range_contrast['range_contrast']:.3f} ± {cl_range_contrast['se']:.3f}")
        print(f"Confidence Interval: {cl_range_contrast['ci_lower']:.3f} - {cl_range_contrast['ci_upper']:.3f}")
        print(f"cohens_d={cl_range_contrast['cohens_d']:.3f}")

    def _add_period_segment(self, ax, y_min=None, y_max=None):
        """
        Add a visual segment for the period on the plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw on.
        y_min : float, optional
            Lower y-limit for the segment. If None, uses ax.get_ylim()[0].
        y_max : float, optional
            Upper y-limit for the segment. If None, uses ax.get_ylim()[1].
        """
        if y_min is None or y_max is None:
            ylims = ax.get_ylim()
            if y_min is None:
                y_min = ylims[0]
            if y_max is None:
                y_max = ylims[1]
        y_delta = y_max - y_min
        alpha1 = 0.15
        alpha2 = 0.3
        alpha3 = 0
        draw_rectangle_gradient(ax, 0, y_min, 2, y_delta, color1='#9e0d0d', color2='#9e0d0d', alpha1=alpha1, alpha2=alpha2)
        rect = mpl.patches.Rectangle((2, y_min), 1, y_delta, facecolor='#9e0d0d', alpha=alpha2, lw=0, zorder=-1)
        ax.add_patch(rect)
        draw_rectangle_gradient(ax, 3, y_min, 3, y_delta, color1='#9e0d0d', color2='#9e0d0d', alpha1=alpha2, alpha2=alpha3)

    def plot_physio_model_pred(self, test_data, data_column, hue_var='age',
                               locked_phase='follicular',
                               ax=None, legend=True,
                               plot_period_seg=True,
                               alpha=1,
                               ytick_digits=2,
                               errorbar=False):

        d_range = self.locking_params[locked_phase]['d_range']
        xticks = self.locking_params[locked_phase]['xticks']

        test_data= test_data.copy(deep=True)
        test_data['age'] = test_data['age'].astype(int).astype(str)

        if hue_var == 'age':
            hue_order = self.age_bin_centers.astype(int).astype(str)
            pal = self.palettes[hue_var]
        elif hue_var == 'cl':
            hue_order = self.CL_bins[::-1]
            pal = self.palettes[hue_var][::-1]
        else:
            hue_order = sorted(test_data[hue_var].unique())
            pal = sns.color_palette("tab10", n_colors=len(hue_order))

        with plt.rc_context(rc=self.plotting_params):
            if ax is None:
                f, ax = plt.subplots(figsize=(6, 3.5),
                                     constrained_layout=False)
                setup_axes(ax)

            sns.lineplot(data=test_data, x='d', y=data_column, hue=hue_var, ax=ax, palette=pal,
                         alpha=alpha, errorbar=None, hue_order=hue_order, lw=1.5)
            # Optionally add error bands if errorbar is a tuple/list and test_data contains ci columns
            if errorbar:
                lower_col = f"{data_column}_lower"
                upper_col = f"{data_column}_upper"
                for key, grp in test_data[test_data[hue_var].isin(hue_order)].groupby(hue_var):
                    color = pal[hue_order.index(key)]
                    ax.fill_between(
                        grp['d'],
                        grp[lower_col],
                        grp[upper_col],
                        alpha=0.15,
                        color=color,
                        linewidth=0
                    )


            ax.set_xlabel("Cycle Day")
            if data_column in self.plotting_labels:
                ax.set_ylabel("Pred. " + self.plotting_physio_labels_short_units[data_column])
            ax.set_xticks(xticks)

            if legend:
                add_legend(ax, title=self.plotting_labels[hue_var])
            else:
                ax.legend().remove()

            fixed_yticks(ax, n_digits_input=ytick_digits, symmetrical_around_zero=True, n_ticks=3)


            if plot_period_seg:
                self._add_period_segment(ax)

        return ax

    def plot_gam_biometrics_cl_age(self, gam_data, plot_period_seg=True, errorbar=None, fixed_age=32, fixed_cl=28):

        locked_phase='follicular'

        n_rows = len(self.CORE_BIOMETRICS)
        alpha = 0.85
        with plt.rc_context(rc=self.plotting_params):
            f, axs = plt.subplots(n_rows, 2, dpi=self.PLOT_DPI, figsize=(6, 1.1*n_rows),
                                  gridspec_kw={'wspace': 0.15, 'hspace': 0.25})

            for jj, hue_var in enumerate(['age', 'cl']):
                if hue_var == 'age':
                    d = gam_data[(gam_data.cl == fixed_cl) & (gam_data.age.isin(self.age_bin_centers))]
                elif hue_var == 'cl':
                    d = gam_data[(gam_data.age == fixed_age) & (gam_data.cl.isin(self.CL_bins))]
                for ii, vv in enumerate(self.gam_models.keys()):
                    ax = axs[ii, jj]
                    setup_axes(ax)
                    if jj == 1:
                        ax.set_ylim(axs[ii, 0].get_ylim())
                        ax.set_yticks(axs[ii, 0].get_yticks())

                    data_column = f"{vv}_pred"
                    self.plot_physio_model_pred(d,
                                                data_column=data_column,
                                                hue_var=hue_var,
                                                locked_phase=locked_phase,
                                                ax=ax, legend=False,
                                                ytick_digits=1, errorbar=errorbar,
                                                alpha=alpha,
                                                plot_period_seg=plot_period_seg)

                    ax.set_xticks(np.arange(-7, 36, 7))

                    if jj==0:
                        ax.set_ylabel(self.plotting_physio_labels_short_units[vv])
                        ax.yaxis.set_label_coords(-0.12, 0.5)
                    else:
                        ax.set_ylabel("")
                        ax.set_yticklabels([])
                    if ii < (n_rows - 1):
                        ax.set_xlabel("")
                        ax.set_xticklabels([])

                    if locked_phase=='follicular':
                        ax.set_xlim([-10, 35])
                    elif locked_phase=='luteal':
                        ax.set_xlim([-7, 28])

            l1 = self._add_colorbar_legend(ax=axs[0,0], cmap=self.palettes['age'], levels=self.age_bin_centers,
                                            step=4, y_delta=1.25, position='top', height_factor=0.1)

            l1.set_xticks(np.arange(len(self.age_bin_centers)))
            l1.set_xticklabels(self.age_bin_centers, fontsize=self.plotting_params.get('legend.fontsize'))
            l1.spines['bottom'].set_linewidth(0)
            l1.set_title("Age [years]", fontsize=self.plotting_params.get('legend.fontsize'))
            l1.tick_params(length=1)

            l2 = self._add_colorbar_legend(ax=axs[0,1], cmap=self.palettes['cl'], levels=np.array(self.CL_bins),
                                            step=2, y_delta=1.25, position='top', height_factor=0.1)
            l2.set_xticks(np.arange(len(self.CL_bins)))
            l2.set_xticklabels(self.CL_bins, fontsize=self.plotting_params.get('legend.fontsize'))
            l2.spines['bottom'].set_linewidth(0)
            l2.set_title("Cycle Length [days]", fontsize=self.plotting_params.get('legend.fontsize'))
            l2.tick_params(length=1)

        label_size = self.plotting_params.get('font.size')
        f.text(0.06, 0.95, "a.", ha='left', va='top', fontsize=label_size, fontweight='bold')
        f.text(0.51, 0.95, "b.", ha='left', va='top',
                fontsize=label_size, fontweight='bold')
        return f, axs

    def _add_colorbar_legend(self, ax, cmap, levels, step, y_delta=0.4, position='bottom', height_factor=0.15):
        """
        Add a colorbar legend (either at the bottom or top of the given axes).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to which the legend will be attached.
        cmap : str or Colormap
            The colormap to use.
        levels : array-like
            The levels to display.
        step : int
            Step size for tick labels.
        y_delta : float, optional
            Relative vertical offset for the legend.
        position : str, optional
            'bottom' or 'top' (default: 'bottom')
        """
        p = ax.get_position()
        f = ax.figure
        if position == 'bottom':
            y = p.y0 - y_delta * p.height
        elif position == 'top':
            y = p.y0 + y_delta * p.height
        else:
            raise ValueError("position must be 'bottom' or 'top'")
        l1 = f.add_axes([p.x0, y, p.width, height_factor * p.height])

        # Convert palette to colormap if needed
        if isinstance(cmap, (list, tuple)):
            # If cmap is a list of colors, create a ListedColormap
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(cmap)
        elif isinstance(cmap, str):
            # If it's a string, matplotlib will handle it
            pass

        l1.imshow(np.vstack((levels, levels)), aspect='auto', cmap=cmap)
        setup_axes(l1, spine_list=['bottom'])
        l1.set_xticks(np.arange(1, len(levels), step))
        l1.set_xticklabels(levels[1::step].astype(int))
        l1.yaxis.set_ticks([])                  # Remove y-axis ticks
        l1.yaxis.set_ticklabels([])             # Remove y-axis labels
        l1.grid(False)

        return l1

    def plot_data_x_day_cl(self, data_column, locked_phase='follicular', ax=None, users=None, line_color=None,
                           legend=True, plot_period_seg=True, errorbar=('ci', 95), ytick_digits=2, alpha=1, cl_bins=None, lw=1.5):

        d_range = self.locking_params[locked_phase]['d_range']
        xticks = self.locking_params[locked_phase]['xticks']

        t = self.reference_table[self.reference_table.cycle_day.isin(d_range)]
        if users is not None:
            t = t[t.n_id.isin(users)]
        a = t.groupby(["n_id", "cl_bin", 'cycle_day'], observed=True)[data_column].mean().reset_index()

        pal = self.palettes['cl'][::-1]
        CL_bins = pd.Series(self.CL_bins[::-1])
        if cl_bins is not None:

            cl_idx = CL_bins[CL_bins.isin(cl_bins)].index.values
            CL_bins = CL_bins[cl_idx].values

            if line_color is not None:
                pal = [line_color] * len(cl_idx)
            else:
                pal = [pal[ii] for ii in cl_idx]

            a = a[a.cl_bin.isin(CL_bins)]
            a['cl_bin'] = pd.Categorical(a['cl_bin'], categories=CL_bins, ordered=True)

        with plt.rc_context(rc=self.plotting_params):
            if ax is None:
                f, ax = plt.subplots(figsize=(6, 3.5),
                                     constrained_layout=False)
                setup_axes(ax)

            sns.lineplot(data=a, x='cycle_day', y=data_column, hue="cl_bin", ax=ax, palette=pal,
                         errorbar=errorbar, hue_order=CL_bins, lw=lw, err_kws={'lw': 0, 'alpha': alpha})
            ax.set_xticks(xticks)
            ax.set_xlabel("Cycle Day")
            if data_column in self.plotting_labels:
                ax.set_ylabel(self.plotting_labels[data_column])

            if data_column[:2] in ['m_', 'z_', 'rz']:
                symmetrical_around_zero = True
            else:
                symmetrical_around_zero = False

            fixed_yticks(ax, n_digits_input=ytick_digits, symmetrical_around_zero=symmetrical_around_zero, n_ticks=3)

            if legend:
                self._add_legend_cl(ax)
            else:
                ax.legend().remove()

            if plot_period_seg:
                self._add_period_segment(ax)

    def plot_data_x_day_age(self, data_column, locked_phase='follicular',
                            min_subjects=50, ax=None, legend=True, errorbar=('ci', 95),
                            plot_period_seg=True, ytick_digits=2,
                            alpha=1, lw=1.5):

        d_range = self.locking_params[locked_phase]['d_range']
        xticks = self.locking_params[locked_phase]['xticks']

        t = self.reference_table[self.reference_table.cycle_day.isin(d_range)]

        # group and mean variable x cycle
        a = pd.DataFrame(t.groupby(['n_id', 'cycle_day', 'age_b'], observed=True)[data_column].count())
        a.rename(columns={data_column: 's_counts'}, inplace=True)
        a[data_column] = t.groupby(['n_id', 'cycle_day', 'age_b'], observed=True)[data_column].mean()
        a = a.reset_index()

        # group again across subjects
        b = pd.DataFrame(a.groupby(['cycle_day', 'age_b'], observed=True)['s_counts'].sum())
        b[data_column] = a.groupby(['cycle_day', 'age_b'], observed=True)[data_column].mean()
        b = b[b.s_counts >= min_subjects].reset_index()

        with plt.rc_context(rc=self.plotting_params):
            if ax is None:
                f, ax = plt.subplots(figsize=(6, 3.5),
                                     constrained_layout=False)
                setup_axes(ax)

            sns.lineplot(data=b, x='cycle_day', y=data_column, hue='age_b', ax=ax,
                         palette=self.palettes['age'], errorbar=errorbar, alpha=alpha, lw=lw)
            ax.set_xlabel("Cycle Day")
            ax.set_xticks(xticks)
            if data_column in self.plotting_labels:
                ax.set_ylabel(self.plotting_labels[data_column])


            if data_column[:2] in ['m_', 'z_', 'rz']:
                symmetrical_around_zero = True
            else:
                symmetrical_around_zero = False
            fixed_yticks(ax, n_digits_input=ytick_digits, symmetrical_around_zero=symmetrical_around_zero, n_ticks=3)

            if legend:
                add_legend(ax, title='Age', bbox_to_anchor=[0.9, 0.1, 0.2, 0.8])
            else:
                ax.legend().remove()


            if plot_period_seg:
                self._add_period_segment(ax)

    def plot_biometrics_cl_age(self, locked_phase='follicular', prefix='m',
                       plot_period_seg=True, errorbar=None):

        n_rows = len(self.CORE_BIOMETRICS)
        alpha=0.85

        with plt.rc_context(rc=self.plotting_params):
            f, axs = plt.subplots(n_rows, 2, dpi=self.PLOT_DPI, figsize=(6, 1.1*n_rows),
                                  gridspec_kw={'wspace': 0.15, 'hspace': 0.25})

            for jj, hue_var in enumerate(['age', 'cl']):
                for ii, vv in enumerate(self.CORE_BIOMETRICS):
                    ax= axs[ii, jj]
                    plot_var = f"{prefix}_{vv}"
                    setup_axes(ax)
                    ytick_sig_fig_digits = 1
                    if plot_var in ['pct_skin_temp']:
                        ax.set_ylim([-1.05, 1.05])
                    if hue_var=='age':
                        self.plot_data_x_day_age(plot_var, locked_phase=locked_phase,
                                                ax=ax, legend=False,
                                                ytick_digits=ytick_sig_fig_digits, errorbar=errorbar,
                                                alpha=alpha)
                    elif hue_var=='cl':
                        ax.set_ylim(axs[ii, 0].get_ylim())
                        ax.set_yticks(axs[ii, 0].get_yticks())
                        self.plot_data_x_day_cl(plot_var,ax=ax,
                                                locked_phase=locked_phase,
                                                legend=False,
                                                ytick_digits=ytick_sig_fig_digits, errorbar=errorbar,
                                                alpha=alpha)

                    if jj==0:
                        ax.set_ylabel(self.plotting_physio_labels_short_units[plot_var])
                        ax.yaxis.set_label_coords(-0.12, 0.5)
                    else:
                        ax.set_ylabel("")
                        ax.set_yticklabels([])
                    if ii < (n_rows - 1):
                        ax.set_xlabel("")
                        ax.set_xticklabels([])

                    if locked_phase=='follicular':
                        ax.set_xlim([-10, 35])
                        ax.set_xticks(np.arange(-7, 36, 7))

            l1 = self._add_colorbar_legend(ax=axs[0, 0], cmap=self.palettes['age'], levels=self.age_bin_centers, step=4, y_delta=1.25, position='top', height_factor=0.1)

            l1.set_xticks(np.arange(-0.5, len(self.age_bin_centers)))
            l1.set_xticklabels(self.age_bin_edges, fontsize=self.plotting_params.get('legend.fontsize'))
            l1.spines['bottom'].set_linewidth(0)
            l1.set_title("Age [years]", fontsize=self.plotting_params.get('legend.fontsize'))
            l1.tick_params(length=1)

            l2 = self._add_colorbar_legend(ax=axs[0,1], cmap=self.palettes['cl'], levels=np.array(self.CL_bins),
                                            step=2, y_delta=1.25, position='top', height_factor=0.1)
            l2.set_xticks(np.arange(-0.5, len(self.CL_bins)))
            l2.set_xticklabels(self.CL_bin_edges.astype(int), fontsize=self.plotting_params.get('legend.fontsize'))
            l2.spines['bottom'].set_linewidth(0)
            l2.set_title("Cycle Length [days]", fontsize=self.plotting_params.get('legend.fontsize'))
            l2.tick_params(length=1)

        label_size = self.plotting_params.get('font.size')
        f.text(0.06, 0.95, "a.", ha='left', va='top', fontsize=label_size, fontweight='bold')
        f.text(0.51, 0.95, "b.", ha='left', va='top',
                fontsize=label_size, fontweight='bold')
        return f, axs

    def _add_legend_cl(self, ax):
        h, l = ax.get_legend_handles_labels()
        if len(h) == 0:
            return
        ax.legend().remove()
        f = ax.figure
        l = f.legend(h, l, loc=3, bbox_to_anchor=[1, 0.05, 0.2, 0.8], title="CL", frameon=True, handlelength=1, fancybox=True, labelspacing=0.2)
        l.get_frame().set_linewidth(0)
        l.get_frame().set_facecolor('0.97')
        l.get_frame().set_alpha(0.9)

    def _add_user_level_biometrics(self, prefix=None):
        if prefix is None:
            prefix = ''
            cols = self.CORE_BIOMETRICS
        else:
            cols = [f"{prefix}_{col}" for col in self.CORE_BIOMETRICS]

        # Check if already run
        if all(f"{prefix}{col}" in self.tables['user'].columns for col in cols):
            return

        #biometric_avg = pd.DataFrame(index=self.users, columns=cols)
        for kk in cols:
            self.tables['user'][kk] = self.data.groupby("n_id")[kk].mean()
            self.tables['user'][kk + '_std'] = self.data.groupby("n_id")[kk].std()

        #self.tables['user'] = pd.concat([self.tables['user'], biometric_avg], axis=1)

    # Plotting Routines
    def plot_user_level_biometrics_vs_x(self, x_var='age_b', prefix=None, save_fig=False):
        self._add_user_level_biometrics(prefix=prefix)
        t = self.tables['user']

        with plt.rc_context(rc=self.plotting_params):
            fig, axes = plt.subplots(5, 2, figsize=(6, 7), sharex=True)
            biometrics = [f"{prefix}_{vv}" for vv in self.CORE_BIOMETRICS] if prefix else self.CORE_BIOMETRICS
            std_suffix = '_std'

            if x_var not in self.plotting_labels:
                x_var_core = x_var.replace('_bin', '')
                if x_var_core in self.plotting_labels:
                    x_label = self.plotting_labels[x_var_core]
                else:
                    x_label = ''
            else:
                x_label = self.plotting_labels[x_var]
            x_label = x_label.replace('\n', '')

            for i, biom in enumerate(biometrics):
                setup_axes(axes[i, 0])

                single_var_point_plot(data=t, x_var=x_var, y_var=biom, ax=axes[i, 0], join_points=False, add_counts=(i == 0), ms=6, dy_factor=0.02)
                axes[i, 0].set_ylabel(self.plotting_physio_labels_short_units[biom])
                axes[i, 0].yaxis.set_label_coords(-0.12, 0.5)
                axes[i, 0].set_xlabel('' if i < 4 else x_label)
                fixed_yticks(axes[i, 0], n_ticks=3, n_digits_input=1)

                # Std plot
                setup_axes(axes[i, 1])
                single_var_point_plot(data=t, x_var=x_var, y_var=biom + std_suffix, ax=axes[i, 1], join_points=False, add_counts=(i == 0), ms=6, dy_factor=0.02)
                axes[i, 1].set_ylabel("")
                axes[i, 1].set_xlabel('' if i < 4 else x_label)

                if biom in ['RR','skin_temp', 'm_RR', 'm_skin_temp']:
                    fixed_yticks(axes[i, 1], n_ticks=3, n_digits_input=2)
                else:
                    fixed_yticks(axes[i, 1], n_ticks=3, n_digits_input=1)

                if i == 0:
                    axes[i, 0].set_title('Mean', fontsize=self.plotting_params['axes.labelsize'])
                    axes[i, 1].set_title('Standard Deviation', fontsize=self.plotting_params['axes.labelsize'])

                axes[i, 0].grid(axis='x', visible=False)
                axes[i, 1].grid(axis='x', visible=False)

                if x_var == 'age_b':
                    axes[i, 0].set_xticklabels(self.age_bin_centers2, rotation=45)
                    axes[i, 1].set_xticklabels(self.age_bin_centers2, rotation=45)
                elif x_var == 'BMI_b2':
                    axes[i, 0].set_xticklabels(self.bmi_bin_centers2, rotation=45)
                    axes[i, 1].set_xticklabels(self.bmi_bin_centers2, rotation=45)

            plt.tight_layout()



            return fig, axes


# ---------------------------------------------------------------------------
# PhysioBehavChangeMethods (notebook 05 prerequisite)
# Source: physio_methods.py:3120-end
# ---------------------------------------------------------------------------
class PhysioBehavChangeMethods(PhysioMethods):
    """Methods for analyzing physiological response to behavior changes by
    menstrual phase.
    """

    def __init__(self, pm: PhysioMethods | None = None, *args, **kwargs):
        if pm is None:
            super().__init__(*args, **kwargs)
        else:
            self.__dict__.update(pm.__dict__)

        pm.add_wo_data()
        pm.add_sleep_2_ref()
        pm.add_wo_2_ref()
        pm.add_acute_chronic_behaviors_data()
        pm.add_acute_chronic_behaviors_ref()
        pm.add_seasonalilty_2_ref()
        pm.add_menstrual_phases_ref()

        self.phases = ['premenstrual', 'menstrual', 'postmenstrual']

        _colors = sns.diverging_palette(270, 120, s=75, l=50, center='light', n=5)
        _colors[2] = np.array(_colors[2]) * 0.7
        self.change_colors = dict(zip(['large_decrease', 'decrease', 'no_change', 'increase', 'large_increase'], _colors))

        self.paper_figures_path = config.FIGURES_DIR

        self._define_behav_change_variables()
        self.behav_change_phase_data = {}
        self.behav_change_phase_summary = {}
        self.behav_physio_models = {}

        colors = ['tab:red', 'tab:orange', 'tab:blue']
        self.phase_colors = {k: colors[ii] for ii, k in enumerate(self.phases)}
        self.phase_days = dict(premenstrual=[-7, -1], menstrual=[0, 6], postmenstrual=[7, 13])

    def _define_behav_change_variables(self):
        self.behav_configs = {
            'acr_sleep_dur': {
                'xlabel': 'Sleep Duration Change [%]',
                'ylabel': 'Cycles [1000s]',
                'chronic_var': 'c_sleep_dur_p1',
                'chronic_range': (7, 8),
                'bins': np.arange(-25, 26, 1),
                'hist_bins': np.arange(-20, 21, 1),
                'group_bins': [-25, -10, -5, 5, 10, 25],
                'x_text_locs': [-15, -7.5, 0, 7.5, 15],
                'yticks': [0, 500, 1000, 1500],
                'ylim': [0, 1750],
                'xgridlines': [-10, -5, 5, 10],
                'xticks': [-10, 0, 10],
                'acr_eval_vals': np.arange(-15, 16, 1),
                'model_vals': [-10, -5, 0, 5, 10],
                'label_mapping': {
                    'large_decrease': "[-25,-10]",
                    'decrease': "(-10,-5]",
                    'no_change': "(-5,5]",
                    'increase': "(5,10]",
                    'large_increase': "(10,25]"
                }
            },
            'acr_eTRIMP': {
                'xlabel': 'Workout Load Change [%]',
                'ylabel': 'Cycles [1000s]',
                'chronic_var': 'c_eTRIMP_p1',
                'chronic_range': (120, 180),
                'bins': np.arange(-100, 101, 5),
                'hist_bins': np.arange(-100, 101, 5),
                'group_bins': [-100, -50, -25, 25, 50, 100],
                'x_text_locs': [-75, -37, 0, 37, 75],
                'yticks': [0, 500],
                'ylim': [0, 600],
                'xgridlines': [-100, -50, -25, 25, 50, 100],
                'xticks': [-100, -50, 0, 50, 100],
                'acr_eval_vals': np.arange(-80, 81, 10),
                'model_vals': [-75, -25, 0, 25, 75],
                'label_mapping': {
                    'large_decrease': "[-100,-50]",
                    'decrease': "(-50,-25]",
                    'no_change': "(-25,25]",
                    'increase': "(25,50]",
                    'large_increase': "(50,100]"
                }
            },
        }
        self.behav_change_variables = list(self.behav_configs.keys())

    def get_behav_change_summary_by_phase(self, behav_var='acr_sleep_dur', physio_prefix='pct'):
        self.physio_prefix = physio_prefix
        rt = self.reference_table
        rt['a_delta_cl_ge_3'] = (np.abs(rt.cycle_length - rt.median_cl) >= 3).astype(int)

        if behav_var not in self.behav_change_variables:
            raise ValueError(f"Behavior '{behav_var}' not configured. Available: {list(self.behav_change_variables)}")

        config_b = self.behav_configs[behav_var]

        phase_summaries = {}
        phase_data = pd.DataFrame()
        for phase in self.phases:

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                warnings.simplefilter("ignore", category=RuntimeWarning)

                x = rt[(rt['phase'] == phase)]
                g = x.groupby(['n_id', 'j_cycle_num'], observed=True)
                x = g[behav_var].last().to_frame()

                x[f'chronic_{behav_var}'] = g[config_b['chronic_var']].first()
                x['cycle_length'] = g['length'].first()
                x['a_delta_cl_ge_3'] = g['a_delta_cl_ge_3'].first()

                sl_onset_mean = g['sl_onset'].apply(circmean_day)
                x['sl_onset_mean'] = sl_onset_mean
                x['sl_onset_cos'] = np.cos(2 * np.pi * sl_onset_mean / 24)
                x['sl_onset_sin'] = np.sin(2 * np.pi * sl_onset_mean / 24)
                x['sl_onset_var'] = g['sl_onset'].apply(circvar_day)

                x['sl_dur_mean'] = g['sleep_dur'].mean()
                x['sl_dur_var'] = g['sleep_dur'].var()
                x['sl_dur_lvar'] = np.log2(x['sl_dur_var'])
                x['sl_dur_mean2'] = x['sl_dur_mean'].mean() ** 2
                x['sl_dur_lvar2'] = x['sl_dur_lvar'] ** 2

                x['wo_dur_mean'] = g['total_wo_duration'].mean()
                x['wo_norm_int_mean'] = g['norm_intensity'].mean()
                x['wo_eTRIMP_mean'] = g['eTRIMP'].mean()
                x['wo_eTRIMP_mean2'] = x['wo_eTRIMP_mean'] ** 2
                x['wo_time_cos'] = g['wo_time_cos'].mean()
                x['wo_time_sin'] = g['wo_time_sin'].mean()

                x['day_of_year'] = g['day_of_year'].first()
                x['cos_season'] = np.cos(x['day_of_year'] * 2 * np.pi / 365.25)
                x['sin_season'] = np.sin(x['day_of_year'] * 2 * np.pi / 365.25)

                x['weekend'] = g['weekend'].last()
            for vv in self.CORE_BIOMETRICS:
                x[f'{physio_prefix}_{vv}'] = g[f'{physio_prefix}_{vv}'].last().to_frame()

            if config_b['chronic_range']:
                chronic_min, chronic_max = config_b['chronic_range']
                y = x.loc[(x[f'chronic_{behav_var}'] >= chronic_min) & (x[f'chronic_{behav_var}'] <= chronic_max)]
            else:
                y = x.copy()

            y[f'{behav_var}_bin'] = pd.cut(y[behav_var], bins=config_b['bins'], include_lowest=True)
            y[f'{behav_var}_bin_groups'] = pd.cut(y[behav_var], bins=config_b['group_bins'],
                                                  labels=['large_decrease', 'decrease', 'no_change',
                                                          'increase', 'large_increase'], include_lowest=True)

            y[f'{behav_var}_bin_locs'] = y[f'{behav_var}_bin_groups'].map({
                k: val for k, val in zip(['large_decrease', 'decrease', 'no_change', 'increase', 'large_increase'], config_b['model_vals'])
            })

            y['phase'] = phase
            y.reset_index(inplace=True)
            phase_data = pd.concat([phase_data, y], ignore_index=True)

            group_summaries = {}
            total_cycles = len(y)

            for group, data in y.groupby(f'{behav_var}_bin_groups', observed=True):
                n_subjects = data.groupby('n_id').size().shape[0]
                n_cycles = data.shape[0]
                pct_cycles = n_cycles / total_cycles * 100 if total_cycles > 0 else 0

                group_summaries[group] = {
                    'n_subjects': n_subjects,
                    'n_cycles': n_cycles,
                    'pct_cycles': pct_cycles,
                    'mean_change': data[behav_var].mean(),
                    'median_change': data[behav_var].median(),
                    'std_change': data[behav_var].std(),
                    'physio_var_summary': {
                        vv: {
                            'mean': data[f'{physio_prefix}_{vv}'].mean(),
                            'median': data[f'{physio_prefix}_{vv}'].median(),
                            'std': data[f'{physio_prefix}_{vv}'].std()
                        }
                        for vv in self.CORE_BIOMETRICS
                    }
                }

            phase_summaries[phase] = {
                'total_cycles': total_cycles,
                'total_subjects': y.groupby('n_id').size().shape[0],
                'mean_change': y[behav_var].mean(),
                'median_change': y[behav_var].median(),
                'std_change': y[behav_var].std(),
                'physio_var_summary': {
                        vv: {
                            'mean': y[f'{physio_prefix}_{vv}'].mean(),
                            'median': y[f'{physio_prefix}_{vv}'].median(),
                            'std': y[f'{physio_prefix}_{vv}'].std()
                        }
                        for vv in self.CORE_BIOMETRICS
                    },
                'by_group': group_summaries
            }

        phase_data['age'] = phase_data.n_id.map(self.tables['user'].age)
        phase_data['age2'] = phase_data['age'] ** 2
        phase_data['BMI'] = phase_data.n_id.map(self.tables['user'].BMI)
        phase_data['BMI2'] = phase_data['BMI'] ** 2
        phase_data['large_decrease'] = (phase_data[f'{behav_var}_bin_groups'] == 'large_decrease').astype(int)
        phase_data['large_increase'] = (phase_data[f'{behav_var}_bin_groups'] == 'large_increase').astype(int)
        phase_data['increase'] = (phase_data[f'{behav_var}_bin_groups'] >= 'increase').astype(int)
        phase_data['decrease'] = (phase_data[f'{behav_var}_bin_groups'] <= 'decrease').astype(int)

        phase_data['weights'] = phase_data[f'{behav_var}_bin_groups'].map(1 / np.log(phase_data[f'{behav_var}_bin_groups'].value_counts())).astype(float)
        phase_data[f"{behav_var}2"] = phase_data[f'{behav_var}'] ** 2

        self.behav_change_phase_data[behav_var] = phase_data
        self.behav_change_phase_summary[behav_var] = phase_summaries
        return phase_data, phase_summaries

    def plot_behav_change_x_phase(self, behav_var='acr_sleep_dur', figure_label=None,
                                  save=False, filename_prefix=None, fig=None, axes=None):
        rt = self.reference_table
        if behav_var not in self.behav_change_variables:
            raise ValueError(f"Behavior '{behav_var}' not configured. Available: {list(self.behav_change_variables)}")

        if behav_var not in self.behav_change_phase_data:
            self.get_behav_change_summary_by_phase(behav_var=behav_var)

        config_b = self.behav_configs[behav_var]

        data = self.behav_change_phase_data[behav_var]
        summary = self.behav_change_phase_summary[behav_var]
        with plt.rc_context(rc=self.plotting_params):

            if fig is None:
                fig = plt.figure(figsize=(6, 1.2), dpi=self.plotting_params.get('figure.dpi'))

                x0 = 0.07
                wspace = 0.03
                width = (1 - x0 - 2 * wspace) / 3
                y0 = 0.15
                height = 0.8
                axes = [
                    fig.add_axes([x0, y0, width, height]),
                    fig.add_axes([x0 + width + wspace, y0, width, height]),
                    fig.add_axes([x0 + 2 * width + 2 * wspace, y0, width, height]),
                ]
            else:
                assert axes is not None, "If 'f' is provided, 'axes' must also be provided."
                assert len(axes) == len(self.phases), "Number of axes must match number of phases."

            legend_fontsize = self.plotting_params.get('legend.fontsize')
            for ii, ax in enumerate(axes):

                setup_axes(ax)
                phase = self.phases[ii]

                phase_data = data[data['phase'] == phase]
                phase_summary = summary[phase]
                cnt = 0

                for group, group_data in phase_data.groupby(f'{behav_var}_bin_groups', observed=True):
                    alpha = 1
                    sns.histplot(group_data[behav_var].dropna(), bins=config_b['hist_bins'], ax=ax,
                                 color=self.change_colors[group], label=group, alpha=alpha)

                    n_subjects = phase_summary['by_group'][group]['n_subjects']
                    n_cycles = phase_summary['by_group'][group]['n_cycles']
                    pct_cycles = n_cycles / phase_summary['total_cycles'] * 100 if phase_summary['total_cycles'] > 0 else 0
                    text_color = self.change_colors[group]
                    if group == 'no_change':
                        text_color = text_color * 0.75
                    elif '_' not in group:
                        text_color = np.array(text_color) * 0.75

                    ax.text(config_b['x_text_locs'][cnt], 1,
                            f'{pct_cycles:0.1f}%', ha='center',
                            va='bottom',
                            fontsize=legend_fontsize, color=text_color, transform=ax.get_xaxis_transform())

                    line_y = 1.15
                    x_delta = (config_b['hist_bins'][-1] - config_b['hist_bins'][0]) * 0.06
                    ax.plot(
                        [config_b['x_text_locs'][cnt] - x_delta, config_b['x_text_locs'][cnt] + x_delta],
                        [line_y, line_y],
                        color=self.change_colors[group],
                        linewidth=3,
                        solid_capstyle='round',
                        zorder=10,
                        transform=ax.get_xaxis_transform(),
                        clip_on=False
                    )
                    cnt += 1

                ax.set_xlabel(config_b['xlabel'])
                ax.set_yticks(config_b['yticks'])
                ax.set_ylim(config_b['ylim'])

                for artist in ax.get_children():
                    if hasattr(artist, 'set_clip_on'):
                        artist.set_clip_on(False)

                if ii == 0:
                    ax.set_ylabel(config_b['ylabel'])
                    ax.set_yticklabels(np.around(np.array(config_b['yticks']) / 1000, 1))
                else:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])

                for gridline in ax.get_ygridlines():
                    gridline.set_visible(False)

                for gridline in ax.get_xgridlines():
                    gridline.set_visible(False)

                for grid_x in config_b['xgridlines']:
                    ax.axvline(grid_x, color='k', linestyle='-', linewidth=0.5, zorder=0, alpha=0.2)

                ax.set_xticks(config_b['xticks'])
                ax.set_xticklabels(config_b['xticks'])

                ax.text(
                    0.5, 1.2, phase.capitalize(),
                    ha='center', va='bottom',
                    fontsize=self.plotting_params.get('axes.labelsize'),
                    transform=ax.transAxes
                )

            if figure_label is not None:
                label_size = self.plotting_params.get('font.size')
                label_weight = 'bold'
                fig.text(0, 1, figure_label, ha="right", va="bottom", fontsize=label_size,
                         fontweight=label_weight, transform=fig.transFigure)

            if save:
                if filename_prefix is None:
                    behavior_name = behav_var.replace('acr_', '')
                    filename_prefix = f'fig4a_{behavior_name}_change_histograms'
                fig.savefig(self.paper_figures_path / f'{filename_prefix}.png',
                            bbox_inches='tight', dpi=self.PLOT_DPI)
                fig.savefig(self.paper_figures_path / f'{filename_prefix}.svg',
                            bbox_inches='tight', dpi=self.PLOT_DPI)

            return fig, axes

    # =========================================================================
    # Phase E-2: per-phase biometric × behavior-change plots
    # =========================================================================

    def add_acr_labels_to_rt(self):
        """Add behavior change labels to the reference table."""
        rt = self.reference_table
        rtg = rt.groupby(['n_id', 'j_cycle_num', 'phase'])

        for behav in self.behav_change_variables:
            cfg = self.behav_configs[behav]

            chronic_var = cfg['chronic_var']
            chronic_range = cfg['chronic_range']
            group_bins = cfg['group_bins']

            binned_acr = pd.cut(
                rtg[behav].last(),
                bins=group_bins,
                include_lowest=True,
                labels=list(cfg['label_mapping'].keys())
            )
            chronic_vals = rtg[chronic_var].first()

            if chronic_range is not None:
                binned_acr[(chronic_vals < chronic_range[0]) | (chronic_vals > chronic_range[1])] = np.nan

            unstacked_binned_acr = binned_acr.unstack()

            for phase in self.phases:
                rt[f'{behav}_bin_groups_{phase}'] = unstacked_binned_acr[phase].reindex(rt.set_index(['n_id', 'j_cycle_num']).index).values

    def plot_physio_behav_change_phase_single_val(self, data, physio_var, ax=None, color='k',
                                                  add_phase_segment=False, legend=False, phase=None,
                                                  phase_color=None, **legend_kwargs):
        """Auxiliary: plot physiological variable response for a single behavior-change group."""
        if ax is None:
            f, ax = plt.subplots(figsize=(5, 3))
            setup_axes(ax)

        sns.lineplot(data=data, x='cycle_day', y=physio_var, color=color,
                     ax=ax, err_kws={'lw': 0})

        ax.set_xlabel("Cycle Day")
        if physio_var in self.plotting_physio_labels_units:
            ax.set_ylabel(self.plotting_physio_labels_units[physio_var])
        elif physio_var in self.plotting_physio_labels:
            ax.set_ylabel(self.plotting_physio_labels[physio_var])

        if legend:
            pass  # add_legend helper not ported; not used by Fig 4b
        else:
            if ax.get_legend():
                ax.legend().remove()

        if add_phase_segment and phase is not None:
            self._add_phase_segment(ax, phase=phase, alpha=0.1, phase_color=phase_color)

        ax.set_xticks(np.arange(-14, 22, 7))

        return ax

    def plot_physio_behav_change_by_phase(self, behav_var='acr_sleep_dur', physio_var='RHR',
                                          physio_prefix='pct', d_range=None, add_phase_title=False,
                                          figure_label=None, save=False, filename_prefix=None,
                                          phase_colors=None, fig=None, axes=None):
        """Plot physiological response to behavior changes across cycle phases (Fig 4b)."""
        if phase_colors is None:
            phase_colors = {
                'premenstrual': "#FFC861",
                'menstrual': "#FFC861",
                'postmenstrual': "#FFC861"
            }

        if d_range is None:
            d_range = np.arange(-15, 22)

        if f'{behav_var}_bin_groups_{self.phases[0]}' not in self.reference_table.columns:
            self.add_acr_labels_to_rt()

        rt = self.reference_table
        yd = rt[(rt['cycle_day'] >= d_range[0]) & (rt['cycle_day'] <= d_range[-1])]

        physio_var_full = f'{physio_prefix}_{physio_var}'
        change_vals = ['large_decrease', 'decrease', 'no_change', 'increase', 'large_increase']

        ylim_config = {
            'pct_RHR': {'ylim': [-5.7, 5.7], 'yticks': [-5, 0, 5]},
            'pct_RR': {'ylim': [-2, 2], 'yticks': [-2, 0, 2]},
            'pct_skin_temp': {'ylim': [-1, 1], 'yticks': [-1, 0, 1]},
            'pct_HRV': {'ylim': [-15, 15], 'yticks': [-15, 0, 15]},
            'pct_blood_oxygen': {'ylim': [-0.5, 0.5], 'yticks': [-0.5, 0, 0.5]}
        }

        with plt.rc_context(rc=self.plotting_params):
            if fig is None:
                fig = plt.figure(figsize=(6, 1.2), dpi=self.plotting_params.get('figure.dpi'))

                x0 = 0.07
                wspace = 0.03
                width = (1 - x0 - 2 * wspace) / 3
                y0 = 0.15
                height = 0.8
                axes = [
                    fig.add_axes([x0, y0, width, height]),
                    fig.add_axes([x0 + width + wspace, y0, width, height]),
                    fig.add_axes([x0 + 2 * width + 2 * wspace, y0, width, height]),
                ]
            else:
                assert axes is not None, "If 'fig' is provided, 'axes' must also be provided."
                assert len(axes) == len(self.phases), "Number of axes must match number of phases."

            for ii, ax in enumerate(axes):
                setup_axes(ax)
                phase = self.phases[ii]
                hue_var = f'{behav_var}_bin_groups_{phase}'

                for kk, change_val in enumerate(change_vals):
                    d = yd[yd[hue_var] == change_val]
                    if len(d) > 0:
                        self.plot_physio_behav_change_phase_single_val(
                            data=d,
                            physio_var=physio_var_full,
                            ax=ax,
                            color=self.change_colors[change_val],
                            legend=False,
                            add_phase_segment=False,
                            phase=phase
                        )

                        if hasattr(self, 'phase_days') and phase in self.phase_days:
                            day = self.phase_days[phase][1]
                            val = d.loc[d['cycle_day'] == day, physio_var_full].mean()
                            if not np.isnan(val):
                                ax.scatter(day, val, color=self.change_colors[change_val],
                                           marker='o', s=20, zorder=10)

                self._add_phase_segment(ax, phase=phase, alpha=0.1, phase_color=phase_colors[phase])

                if physio_var_full in ylim_config:
                    ax.set_ylim(ylim_config[physio_var_full]['ylim'])
                    ax.set_yticks(ylim_config[physio_var_full]['yticks'])

                if ii != 0:
                    ax.set_yticklabels([])
                    ax.set_ylabel('')
                else:
                    yticklabels = [f"{tick:.1f}" for tick in ax.get_yticks()]
                    ax.set_yticklabels(yticklabels)
                    ax.set_ylabel(f"{self.plotting_physio_labels_short[physio_var]} [%]")

                if add_phase_title:
                    ax.text(
                        0.5, 1.05, phase.capitalize(),
                        ha='center', va='bottom',
                        fontsize=self.plotting_params.get('font.size') * 1.1,
                        fontweight='bold',
                        transform=ax.transAxes
                    )

                for gridline in ax.get_xgridlines():
                    if not np.isclose(gridline.get_xdata()[0], 0):
                        gridline.set_visible(False)
                    else:
                        gridline.set_linewidth(1.1)

            if figure_label is not None:
                label_size = self.plotting_params.get('font.size')
                label_weight = 'bold'
                fig.text(0, 1, figure_label, ha="left", va="top",
                         fontsize=label_size, fontweight=label_weight,
                         transform=fig.transFigure)

        return fig, axes
