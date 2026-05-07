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

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from tqdm import tqdm

from . import config
from ._plot_utils import setup_axes
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
