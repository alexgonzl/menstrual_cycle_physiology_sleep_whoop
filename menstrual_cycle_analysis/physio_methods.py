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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from tqdm import tqdm

from .cl_behav_methods import CycleBehavMethods


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
