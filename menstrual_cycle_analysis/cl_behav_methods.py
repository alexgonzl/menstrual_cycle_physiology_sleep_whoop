"""`CycleBehavMethods` — verbatim port of the source class with notebook 01
scope kept (Fig 1 + S1-S4). Methods unused by notebook 01 (phase tables,
bootstrapped onset comparisons, alternative population-averaged contrasts,
the older `get_CL_summaries` code path) are intentionally not included; they
will be ported when the notebooks that need them are added.

Source: `whoop_analyses/whoop_analyses/cl_behav_methods.py` (CycleBehavMethods).

Modifications from source:
  - Constructor takes (data, summary_df, cycle_type='j') instead of pulling
    `WHOOP_USER_TABLES`. The summary table is the public CSV `power_users_summary_table.csv`.
  - `self.wt.HR_ZONES` / `self.wt.HR_WEIGHTS` are replaced with
    `config.HR_ZONES` / `config.HR_WEIGHTS`.
  - `self.wt.aggregate_table[kk]` (used to attach age, BMI) is replaced
    with `self.summary_df[kk]`.
  - `self.wt.activity_categories` block in `_add_workout_categories` is dropped
    (activity counts aren't used by notebook 01).
  - Method bodies are otherwise byte-identical to the source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from ._plot_utils import (
    fixed_yticks,
    get_plotting_params,
    setup_axes,
    single_var_point_plot,
)
from .stats.circular import circmean_day, circvar_day, mad


class CycleBehavMethods:

    MIN_CL = config.MIN_CL
    MAX_CL = config.MAX_CL
    MIN_MEDIAN_CL = config.MIN_MEDIAN_CL
    MAX_MEDIAN_CL = config.MAX_MEDIAN_CL
    MIN_N_VCL = config.MIN_N_VCL

    SL_DUR_CAT_EDGES = config.SL_DUR_CAT_EDGES
    SL_DUR_CAT_LABELS = config.SL_DUR_CAT_LABELS
    SL_DUR_BIN_EDGES = config.SL_DUR_BIN_EDGES
    SL_DUR_BIN_LABELS = config.SL_DUR_BIN_LABELS
    SL_ONSET_EDGES = config.SL_ONSET_BIN_EDGES
    SL_DUR_LVAR_BINS = config.SL_DUR_LVAR_BINS
    SL_DUR_VAR_BINS = config.SL_DUR_VAR_BINS
    SL_DUR_SD_BINS = config.SL_DUR_SD_BINS

    SL_ONSET_CAT = config.SL_ONSET_BIN_LABELS

    Z_DAYS_LEVELS = [0, 1, 2, 3]

    PHASE_LENGTH_DAYS = config.PHASE_LENGTH_DAYS
    MAX_DAY_DEV = config.MAX_DAY_DEV

    MIN_WO_INTENSITY = config.MIN_WO_INTENSITY

    PLOT_DPI = 500

    def __init__(self, data, summary_df, cycle_type='j'):
        self.summary_df = summary_df

        self.data = data.copy(deep=True)
        self.cycle_type = cycle_type
        if cycle_type == 'j':
            self.add_jc_column()

        self.data['day_of_year'] = pd.DatetimeIndex(self.data.date).dayofyear
        self._c_col = f"{cycle_type}_cycle"
        self._cn_col = f"{cycle_type}_cycle_num"
        self._cd_col = f"{cycle_type}_cycle_day"
        self._cp_col = f'{cycle_type}_CP'
        self._cb_col = f'{cycle_type}_bounds'

        self._variable_bins()
        self._plotting_labels()

        self._add_cycle_info_to_data()
        self.drop_unbounded_cycles()
        try:
            self._add_workout_categories()
        except Exception:
            pass

        self.tables = dict(user=None, cycle=None, phase=None)
        self.get_user_table()
        self.get_cycle_table()

        self.plotting_params = get_plotting_params()
        self.age_axes_width = 4

    # ------------------------------------------------------------------
    # Plotting labels and bin definitions
    # ------------------------------------------------------------------
    def _plotting_labels(self):
        _sleep_vars_labels = dict(sleep_dur="Sleep Duration \n[hours]",
                                  sl_dur_mean="Sleep Duration \n[hours]",
                                  sl_dur_bin="Sleep Duration \n[hours]",
                                  sl_TiB_mean="Time in Bed \n[hours]",
                                  sl_TiB_bin="Time in Bed \n[hours]",
                                  sl_dur_var="Sleep Duration Var. [au]",
                                  sleep_dur_var="Sleep Duration Var. [au]",
                                  sl_dur_var_bin="Sleep Duration Var. [au]",
                                  sl_onset_mean_b="Sleep Onset",
                                  sl_onset_mean_b2="Sleep Onset",
                                  sl_onset_var="Sleep Onset Var. [au]",
                                  sl_onset_var_bin="Sleep Onset Var. [au]",)

        _wo_vars_labels = dict(total_intensity="Avg. Heart Rate \nIntensity [au]",
                               eTRIMP_bin="Daily Workout Load \n [HR Zone One \n equivalent mins.]",
                               wo_dur_bin="Daily Workout Duration \n [minutes]",
                               total_intensity_eTRIMP="Daily Workout Load \n [HR Zone One \n equivalent mins.]",
                               total_intensity_eTRIMP_cat_num="Daily Workout Load \n [HR Zone One \n equivalent mins.]",
                               daily_activity_rate="Proportion of days with an activity",
                               cardio="Proportion of cardio activities",
                               wo_onset_mean_b="Mean Sleep Onset [Hr]",
                               wo_onset_var="Sleep Onset Var. [au]",)

        _outcome_labels = dict(median_cl='Median Cycle Length [days]',
                               mad_cl="Median Absolute Deviation [days]",
                               sd_cl=f"Cycle Length \n $\\sigma$ [days]",
                               a_delta_cl_ge_3="Probability of \n Cycle Length \nDeviation",
                               prop_a_delta_cl_ge_3=r"$P[|CL - \widebar{CL}| \geq X]$",
                               prop_long=r"$P[|CL - \widebar{CL}| \geq X]$",
                               prop_delta_cl_ge_3=r"$P[|CL - \widebar{CL}| \geq X]$",
                               prop_delta_cl_le_n3=r"$P[|CL - \widebar{CL}| \geq X]$",
                               prop_ge_3_le_3=r"$\frac{P[ \Delta CL \geq 3]}{P[ |\Delta CL| \geq 3]}$")

        _independent_vars_labels = dict(age='Age [years]',
                                        age_b='Age [years]',
                                        BMI='BMI',
                                        BMI_b='BMI',
                                        BMI_b2='BMI')

        self.plotting_labels = _outcome_labels | _independent_vars_labels | _sleep_vars_labels | _wo_vars_labels

    def _variable_bins(self):

        # AGE
        self.age_bin_edges, self.age_bin_centers = self._age_bins()
        self.age_bin_centers = self.age_bin_centers.astype(int)
        _, self.age_bin_centers2 = self._age_bins2()
        self._min_age = self.age_bin_edges[0]
        self._max_age = self.age_bin_edges[-1]

        # BMI
        self.bmi_bin_edges, self.bmi_bin_centers = self._bmi_bins()
        self.bmi_bin_edges2, self.bmi_bin_centers2 = self._bmi_bins2()
        self._min_bmi = self.bmi_bin_edges[0]
        self._max_bmi = self.bmi_bin_edges[-1]

        # CL
        self.delta_cl_bins = [-10, -5.6, -2.6, -0.6, 0.6, 2.6, 5.6, 10]
        self.delta_cl_labels = ['<-6', '-5 to -3', '-2 to -1', '0', '1 to 2', '3 to 5', '>6']
        self.delta_cl_labels2 = ['(-10, 6]', '[-5, -3]', '[-2, -1]', '0', '[1, 2]', '[3, 5]', '[6, 10]']

        # SLEEP
        self.sl_onset_bins = [0, 6, 20, 22, 24]
        self.sl_onset_labels = ['12-6am', '6am-8pm', '8-10pm', '10pm-12am']

        self.sl_dur_cat_bins = [0, 6, 7.5, 12]
        self.sl_dur_cat_labels = ['<=6', '6-7.5', '>7.5']

        self.sl_dur_bins = self.SL_DUR_BIN_EDGES
        labels = [f"({i},{j}]" for i, j in zip(self.SL_DUR_BIN_EDGES[:-1], self.SL_DUR_BIN_EDGES[1:])]
        labels[0] = f"[{self.SL_DUR_BIN_EDGES[0]},{self.SL_DUR_BIN_EDGES[1]}]"
        self.sl_dur_labels = labels
        self.sl_TiB_labels = labels
        self._min_sl = self.SL_DUR_BIN_EDGES[0]
        self._max_sl = self.SL_DUR_BIN_EDGES[-1]

        self.sl_dur_var_bins = [0, 0.5, 1, 2, 4, 8]
        self.sl_dur_var_labels = ['vlow', 'low', 'med', 'high', 'vhigh']

        # WORKOUTS
        self.eTRIMP_bin_edges = np.array([0, 60, 120, 180, 240, 300, 600])
        self.eTRIMP_bin_centers = self.eTRIMP_bin_edges[1:] - np.diff(self.eTRIMP_bin_edges) / 2
        labels = [f"({i},{j}]" for i, j in zip(self.eTRIMP_bin_edges[:-1], self.eTRIMP_bin_edges[1:])]
        labels[0] = f"[{self.eTRIMP_bin_edges[0]},{self.eTRIMP_bin_edges[1]}]"
        self.eTRIMP_bin_labels = labels

        self.wo_dur_bin_edges = np.array([0, 30, 60, 90, 120, 150, 180, 210, 500])
        self.wo_dur_bin_centers = self.wo_dur_bin_edges[1:] - np.diff(self.wo_dur_bin_edges) / 2
        labels = [f"({i},{j}]" for i, j in zip(self.wo_dur_bin_edges[:-1], self.wo_dur_bin_edges[1:])]
        labels[0] = f"[{self.wo_dur_bin_edges[0]},{self.wo_dur_bin_edges[1]}]"
        self.wo_dur_bin_labels = labels

        self.norm_intensity_bins = [0.5, 2.25, 3.5, 5.5]
        self.norm_intensity_labels = [1, 3, 5]
        self.wo_onset_bins = [0, 4, 6, 10, 16, 20, 22, 24]
        self.wo_onset_labels = ['0-4am', '4am-6pm', '6-10am', '10am-4pm', '4-8pm', '8-10pm', '10am-12pm']

    # ------------------------------------------------------------------
    # Cycle column derivation
    # ------------------------------------------------------------------
    def add_jc_column(self):
        if 'j_cycle' in self.data.columns:
            return
        t = self.data
        t['j_cycle'] = t['starts']
        t['j_cycle_num'] = t.groupby("n_id")['j_cycle'].transform(lambda x: (x == 1).cumsum())
        a = t.loc[t.j_cycle == 1, 'day']
        ads = a.diff().shift(-1)
        b = (ads // 2) + a.index
        c = b[(ads >= self.MIN_CL) & (ads <= self.MAX_CL)].values.astype(int)
        t.loc[c, 'j_cycle'] = -1

    def _add_cycle_info_to_data(self):

        data = self.data

        # cycle bounds
        data[self._cb_col] = 0
        dg = data.groupby("n_id")
        for u in data.n_id.unique():
            d = dg.get_group(u)
            d0, dE = d.index[d[self._c_col] == 1][[0, -1]]
            data.loc[np.arange(d0, dE + 1), self._cb_col] = 1

        # cycle day
        df_user_cycle = data.groupby(["n_id", self._cn_col])
        data[self._cd_col] = df_user_cycle['day'].transform(lambda x: np.arange(len(x)))

        # cycle phase
        data[self._cp_col] = 'other'
        data.loc[(data[self._cd_col] >= self.PHASE_LENGTH_DAYS) &
                 (data[self._cd_col] < self.PHASE_LENGTH_DAYS * 2), self._cp_col] = 'post_bleed'
        data.loc[(data[self._cd_col].shift(-self.PHASE_LENGTH_DAYS) < self.PHASE_LENGTH_DAYS), self._cp_col] = 'pre_bleed'
        data.loc[data[self._cd_col] < self.PHASE_LENGTH_DAYS, self._cp_col] = 'bleed'

    def drop_unbounded_cycles(self):
        self.data = self.data[self.data[self._cb_col] == 1]

    def _add_workout_categories(self):
        data = self.data
        temp = pd.DataFrame()
        # workouts
        for ii, kk in enumerate(['wo_time_0', 'wo_time_1', 'wo_time_2']):
            idx = data[f'intensity_{ii}'] < self.MIN_WO_INTENSITY
            data.loc[idx, kk] = np.nan
            temp[kk] = pd.cut(data[kk], [0, 5, 8, 18, 24], labels=['0-5', '5-8', '8-18', '18-24'], include_lowest=True)

        data['wo_time_cos'] = np.cos((data[['wo_time_0', 'wo_time_1', 'wo_time_2']]) * 2 * np.pi / 24).mean(axis=1)
        data['wo_time_sin'] = np.sin((data[['wo_time_0', 'wo_time_1', 'wo_time_2']]) * 2 * np.pi / 24).mean(axis=1)
        data['n_morning_wo'] = (temp[['wo_time_0', 'wo_time_1', 'wo_time_2']] == '5-8').sum(axis=1)
        data['n_night_wo'] = (temp[['wo_time_0', 'wo_time_1', 'wo_time_2']] == '18-24').sum(axis=1)
        data['n_wo'] = (~temp.isna()).sum(axis=1)
        data['total_wo_duration'] = (data[[f"duration_{ii}" for ii in range(3)]].sum(axis=1)) * 60
        data['total_intensity_eTRIMP'] = (data[config.HR_ZONES] * config.HR_WEIGHTS).sum(axis=1)

    def _add_behav_cat(self, df):
        columns = df.columns
        if 'sl_onset_mean' in columns:
            df['sl_onset_cat'] = pd.cut(df.sl_onset_mean, self.SL_ONSET_EDGES, labels=self.SL_ONSET_CAT, include_lowest=True)
        if 'sl_dur_mean' in columns:
            df['sl_dur_bin'] = pd.cut(df.sl_dur_mean, self.SL_DUR_BIN_EDGES, labels=self.SL_DUR_BIN_LABELS, include_lowest=True)
        if 'sl_TiB_mean' in columns:
            df['sl_TiB_bin'] = pd.cut(df.sl_TiB_mean, self.SL_DUR_BIN_EDGES, labels=self.SL_DUR_BIN_LABELS, include_lowest=True)
        if 'wo_time_index' in columns:
            df['wo_time_cat'] = pd.cut(df.wo_time_index, [-1, -0.2, 0.2, 1], labels=['evening', 'neutral', 'morning'], include_lowest=True)
        if 'total_wo_duration' in columns:
            df['wo_dur_bin'] = pd.cut(df.total_wo_duration, self.wo_dur_bin_edges, labels=self.wo_dur_bin_labels, include_lowest=True)
        if 'norm_intensity' in columns:
            df['wo_norm_int_bin'] = pd.cut(df.norm_intensity, self.norm_intensity_bins, labels=self.norm_intensity_labels, include_lowest=True)
        if 'total_intensity_eTRIMP' in columns:
            df['eTRIMP_bin'] = pd.cut(df.total_intensity_eTRIMP, self.eTRIMP_bin_edges, labels=self.eTRIMP_bin_labels, include_lowest=True)
        return df

    # ------------------------------------------------------------------
    # User and cycle tables
    # ------------------------------------------------------------------
    def get_user_table(self):
        data = self.data

        cn_col = self._cn_col

        df_user_cycle = data.groupby(["n_id", cn_col], observed=True)

        # summary measures by cycle and user
        cl_x_user_x_cycle = df_user_cycle['date'].apply(len)
        vcl_x_user_x_cycle = (cl_x_user_x_cycle >= self.MIN_CL) & (cl_x_user_x_cycle <= self.MAX_CL)

        user_table = pd.DataFrame(index=data.n_id.unique())
        user_table['n_c'] = cl_x_user_x_cycle.groupby('n_id').count()
        user_table['n_vc'] = cl_x_user_x_cycle[vcl_x_user_x_cycle].groupby('n_id').count()
        user_table['length'] = data.groupby('n_id')['date'].apply(len)

        # summary measures by user
        user_table['mean_cl'] = cl_x_user_x_cycle[vcl_x_user_x_cycle].groupby('n_id').mean()
        user_table['sd_cl'] = cl_x_user_x_cycle[vcl_x_user_x_cycle].groupby('n_id').std()

        user_table['median_cl'] = cl_x_user_x_cycle[vcl_x_user_x_cycle].groupby('n_id').median()
        user_table['mad_cl'] = cl_x_user_x_cycle[vcl_x_user_x_cycle].groupby('n_id').apply(mad)

        user_table['valid_user'] = (user_table.median_cl >= self.MIN_MEDIAN_CL) & (user_table.median_cl <= self.MAX_MEDIAN_CL) & (user_table['n_vc'] >= self.MIN_N_VCL)

        user_table['E_n_c'] = np.round(user_table['length'] / user_table['mean_cl'])

        user_table = user_table[user_table.valid_user]

        for kk in ['age', 'BMI']:
            user_table[kk] = user_table.index.map(self.summary_df[kk])
            user_table[f'{kk}2'] = user_table[kk] ** 2

        user_table['age_b'] = pd.cut(user_table.age, self.age_bin_edges, labels=self.age_bin_centers, include_lowest=True)
        user_table['BMI_b'] = pd.cut(user_table.BMI, self.bmi_bin_edges, labels=self.bmi_bin_centers, include_lowest=True)
        user_table['BMI_b2'] = pd.cut(user_table.BMI, self.bmi_bin_edges2, labels=self.bmi_bin_centers2, include_lowest=True)

        # store results
        self.tables['user'] = user_table
        self.users = user_table.index
        self.data = data[data.n_id.isin(self.users)]

    def get_cycle_table(self):
        if self.tables['user'] is None:
            self.get_user_table()

        if self.tables['cycle'] is not None:
            return self.tables['cycle']

        data = self.data
        user_table = self.tables['user']
        cn_col = self._cn_col

        df_user_cycle = data.groupby(["n_id", cn_col], observed=True)

        # summary measures by cycle and user
        cl_x_user_x_cycle = df_user_cycle['day'].apply(len)
        vcl_x_user_x_cycle = (cl_x_user_x_cycle >= self.MIN_CL) & (cl_x_user_x_cycle <= self.MAX_CL)

        # cycle table
        cycle_table = pd.DataFrame(cl_x_user_x_cycle)
        cycle_table.rename(columns={'day': 'length'}, inplace=True)
        cycle_table['vcl'] = vcl_x_user_x_cycle

        # these are by-user metrics that don't change by cycle
        metrics = ['median_cl', 'mean_cl', 'mad_cl', 'sd_cl']
        for metric in metrics:
            cycle_table[metric] = cycle_table.index.get_level_values('n_id').map(user_table[metric])

        # by cycle table delta CLs
        cycle_table['delta_cl'] = cycle_table['length'] - cycle_table['median_cl']
        cycle_table['z_delta_cl'] = (cycle_table['length'] - cycle_table['mean_cl']) / cycle_table['sd_cl']

        # filter data for valid cycles UPDATED on 5/6/2025
        cycle_table = cycle_table[cycle_table.vcl]

        # by cycle delta thresholds
        self.delta_cl_thr_cols = []
        for thr in np.arange(0, self.MAX_DAY_DEV, 1):
            col = f'a_delta_cl_ge_{thr}'
            y = np.abs(cycle_table['delta_cl']) >= thr
            cycle_table[col] = y.astype(int)
            user_table[col] = y.groupby(level='n_id').mean()
            self.delta_cl_thr_cols.append(col)

        self.z_delta_cl_thr_cols = []
        for thr in self.Z_DAYS_LEVELS:
            col = f'a_z_delta_cl_ge_{thr}'
            y = np.abs(cycle_table['z_delta_cl']) >= thr
            cycle_table[col] = y
            user_table[col] = y.groupby(level='n_id').mean()
            self.z_delta_cl_thr_cols.append(col)

        # seasonality
        cycle_table['data_day'] = df_user_cycle['day'].first()
        cycle_table['day_of_year'] = df_user_cycle['day_of_year'].first()
        cycle_table['cycle_length'] = cycle_table['length']
        cycle_table['cos_season'] = np.cos(2 * np.pi * cycle_table['day_of_year'] / 365.25)
        cycle_table['sin_season'] = np.sin(2 * np.pi * cycle_table['day_of_year'] / 365.25)

        self._add_age_bmi_to_sub_tables(cycle_table)

        # weights
        for kk in ['age_b', 'BMI_b', 'BMI_b2']:
            cycle_table = self._compute_weights_for_bins(cycle_table, kk)

        self.tables['cycle'] = cycle_table
        self.tables['user'] = user_table

        return cycle_table

    def _compute_weights_for_bins(self, table, col):
        weights_col = f'{col}_weights'
        # Remove '_bin' from the column name if present
        if col.endswith('_bin'):
            raw_col = col.replace('_bin', '')
        elif col.endswith('_b'):
            raw_col = col.replace('_b', '')
        elif col.endswith('_b2'):
            raw_col = col.replace('_b2', '')
        elif col.endswith('_cat'):
            raw_col = col.replace('_cat', '')
        elif col.endswith('_cat_num'):
            raw_col = col.replace('_cat_num', '')
        else:
            raw_col = col

        found_boundaries = True
        if hasattr(table[col].cat.categories, 'left'):
            lowest_bin_edge = table[col].cat.categories[0].left
            highest_bin_edge = table[col].cat.categories[-1].right
        else:
            if raw_col == 'age':
                lowest_bin_edge = self._min_age
                highest_bin_edge = self._max_age
            elif raw_col == 'BMI':
                lowest_bin_edge = self._min_bmi
                highest_bin_edge = self._max_bmi
            elif raw_col == 'sl_onset':
                lowest_bin_edge = 0
                highest_bin_edge = 24
            elif raw_col in ['sl_dur', 'sl_TiB', 'sl_dur_mean', 'sl_TiB_mean', 'sleep_dur']:
                lowest_bin_edge = self._min_sl
                highest_bin_edge = self._max_sl
            else:
                found_boundaries = False

        table[weights_col] = table.groupby(col, observed=True)[col].transform(lambda x: 1 / np.log(len(x)) if len(x) > 0 else 0)

        if found_boundaries:
            cats = np.array(table[col].cat.categories)

            # weights for the lowest and highest bins
            low_idx = table[col] == cats[0]
            high_idx = table[col] == cats[-1]
            if low_idx.sum() > 0:
                weight_lowest_bin = table.loc[low_idx, weights_col].iloc[0]
                table.loc[(table[raw_col] < lowest_bin_edge), weights_col] = weight_lowest_bin
            if high_idx.sum() > 0:
                weight_highest_bin = table.loc[high_idx, weights_col].iloc[0]
                table.loc[(table[raw_col] > highest_bin_edge), weights_col] = weight_highest_bin

        return table

    def _add_age_bmi_to_sub_tables(self, table):
        for kk in ['age', 'BMI', 'age2', 'BMI2', 'age_b', 'BMI_b', 'BMI_b2']:
            table[kk] = table.index.get_level_values('n_id').map(self.tables['user'][kk])
        return table

    # ------------------------------------------------------------------
    # Sleep / workout cycle aggregation
    # ------------------------------------------------------------------
    def add_sleep_behaviors(self, level='cycle'):

        df = self.tables[level]

        # Create groupby object once and reuse
        if level == 'user':
            data_grouped = self.data.groupby('n_id')
        elif level == 'cycle':
            data_grouped = self.data.groupby(['n_id', self._cn_col])
        elif level == 'phase':
            data_grouped = self.data.groupby(['n_id', self._cn_col, self._cp_col])

        # Vectorized sleep onset calculations
        sl_onset_mean = data_grouped['sl_onset'].apply(circmean_day)
        df['sl_onset_mean'] = sl_onset_mean
        df['sl_onset_cos'] = np.cos(2 * np.pi * sl_onset_mean / 24)
        df['sl_onset_sin'] = np.sin(2 * np.pi * sl_onset_mean / 24)
        df['sl_onset_var'] = data_grouped['sl_onset'].apply(circvar_day)

        # Vectorized sleep duration calculations
        sl_dur_mean = data_grouped['sleep_dur'].mean()
        sl_dur_var = data_grouped['sleep_dur'].var()
        sl_dur_sd = data_grouped['sleep_dur'].std()

        df['sl_dur_mean'] = sl_dur_mean
        df['sl_dur_var'] = sl_dur_var
        df['sl_dur_sd'] = sl_dur_sd
        df['sl_dur_lvar'] = np.log2(sl_dur_var)
        df['sl_dur_lvar2'] = df['sl_dur_lvar'] ** 2
        df['sl_dur_mean2'] = sl_dur_mean ** 2

        # Vectorized time in bed calculations
        sl_TiB_mean = data_grouped['time_in_bed'].mean()
        sl_TiB_var = data_grouped['time_in_bed'].var()

        df['sl_TiB_mean'] = sl_TiB_mean
        df['sl_TiB_var'] = sl_TiB_var
        df['sl_TiB_lvar'] = np.log2(sl_TiB_var)
        df['sl_TiB_lvar2'] = df['sl_TiB_lvar'] ** 2
        df['sl_TiB_mean2'] = sl_TiB_mean ** 2

        # Efficient binning - use dictionary for batch operations
        binning_operations = {
            'sl_onset_mean_cat': (sl_onset_mean, self.SL_ONSET_EDGES, self.SL_ONSET_CAT),
            'sl_dur_mean_cat': (sl_dur_mean, self.SL_DUR_CAT_EDGES, self.SL_DUR_CAT_LABELS),
            'sl_dur_mean_bin': (sl_dur_mean, self.SL_DUR_BIN_EDGES, self.SL_DUR_BIN_LABELS),
            'sl_dur_var_bin': (sl_dur_var, self.SL_DUR_VAR_BINS, None),
            'sl_dur_lvar_bin': (df['sl_dur_lvar'], self.SL_DUR_LVAR_BINS, None),
            'sl_dur_sd_bin': (sl_dur_sd, self.SL_DUR_SD_BINS, None),
            'sl_TiB_mean_bin': (sl_TiB_mean, self.SL_DUR_BIN_EDGES, self.SL_DUR_BIN_LABELS)
        }

        # Batch binning operations
        bins_to_weight = []
        for col_name, (data_col, bins, labels) in binning_operations.items():
            if labels is not None:
                df[col_name] = pd.cut(data_col, bins, labels=labels, include_lowest=True)
            else:
                df[col_name] = pd.cut(data_col, bins, include_lowest=True)
            bins_to_weight.append(col_name)

        # Batch weight computation
        for col in bins_to_weight:
            df = self._compute_weights_for_bins(df, col)

        return df

    def add_workout_behaviors(self, level='cycle'):

        df = self.tables[level]

        if level == 'user':
            data_x_level = self.data.groupby('n_id')
        elif level == 'cycle':
            data_x_level = self.data.groupby(['n_id', f'{self.cycle_type}_cycle_num'])
        elif level == 'phase':
            data_x_level = self.data.groupby(['n_id', f'{self.cycle_type}_cycle_num', f'{self.cycle_type}_CP'])

        # WORKOUTS
        df['n_morning_wo'] = data_x_level['n_morning_wo'].sum()
        df['n_night_wo'] = data_x_level['n_night_wo'].sum()
        df['n_wo'] = data_x_level['n_wo'].sum()
        df['wo_time_index'] = (df['n_morning_wo'] - df['n_night_wo']) / (df['n_morning_wo'] + df['n_night_wo'])
        df['wo_time_cat'] = pd.cut(df['wo_time_index'], [-1, -0.2, 0.2, 1], labels=['evening', 'neutral', 'morning'], include_lowest=True)
        df['mo_wo'] = df.wo_time_cat.map(dict(evening=-1, neutral=0, morning=1)).astype(float)
        df['wo_rate'] = df['n_wo'] / df['length']
        df['wo_time_cos'] = data_x_level['wo_time_cos'].mean()
        df['wo_time_sin'] = data_x_level['wo_time_sin'].mean()

        df['total_wo_duration'] = data_x_level['total_wo_duration'].mean()
        df[config.HR_ZONES] = data_x_level[config.HR_ZONES].mean()
        df['total_intensity_eTRIMP'] = data_x_level['total_intensity_eTRIMP'].mean()
        df['eTRIMP'] = df['total_intensity_eTRIMP']
        df['eTRIMP2'] = df['total_intensity_eTRIMP'] ** 2
        df['norm_intensity'] = df['total_intensity_eTRIMP'] / df['total_wo_duration']
        df['norm_intensity2'] = df['norm_intensity'] ** 2

        df['eTRIMP_bin'] = pd.cut(df.total_intensity_eTRIMP, bins=self.eTRIMP_bin_edges, labels=self.eTRIMP_bin_labels, include_lowest=True)
        df['wo_dur_bin'] = pd.cut(df.total_wo_duration, self.wo_dur_bin_edges, labels=self.wo_dur_bin_labels, include_lowest=True)
        df['wo_norm_int_bin'] = pd.cut(df.norm_intensity, self.norm_intensity_bins, labels=self.norm_intensity_labels, include_lowest=True)

        df = self._compute_weights_for_bins(df, 'eTRIMP_bin')

        return df

    # ------------------------------------------------------------------
    # Plot helpers (used by CycleLengthAnalyses for Fig 1, S1-S4)
    # ------------------------------------------------------------------
    def plot_var_x_age(self, y_var, data=None, ax=None, yticks=None, n_digits=None, ylims=None, **kwargs):
        import matplotlib.pyplot as plt

        if data is None:
            data = self.tables['user']

        buffer = 0.1
        with plt.rc_context(rc=self.plotting_params):
            if ax is None:
                f, ax = plt.subplots(figsize=(self.age_axes_width,
                                              self.age_axes_width * 4 / 5),
                                     constrained_layout=False)
                ax.set_position([0.15, 0.15, 0.7, 0.7])
            else:
                f = ax.figure
            setup_axes(ax)

            single_var_point_plot(data=data, x_var='age_b', y_var=y_var, ax=ax, **kwargs)
            if y_var in self.plotting_labels:
                ax.set_ylabel(self.plotting_labels[y_var])
            ax.set_xlabel("Age")
            xticks = np.arange(len(self.age_bin_centers))
            ax.set_xticks(xticks)
            labels = self.age_bin_centers2[xticks]
            ax.set_xticklabels(labels, rotation=45)
            if yticks is None:
                fixed_yticks(ax, buffer=buffer, n_digits_input=n_digits)
            else:
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticks)
                if ylims is not None:
                    ax.set_ylim(ylims)
                else:
                    delta = (yticks[-1] - yticks[0])
                    ax.set_ylim(yticks[0] - buffer * delta, yticks[-1] + buffer * delta)

        return f, ax

    def plot_var_x_BMI(self, y_var, data=None, ax=None, yticks=None, n_digits=None, ylims=None, **kwargs):
        import matplotlib.pyplot as plt

        if data is None:
            data = self.tables['user']

        buffer = 0.1
        with plt.rc_context(rc=self.plotting_params):
            if ax is None:
                f, ax = plt.subplots(figsize=(self.age_axes_width,
                                              self.age_axes_width * 4 / 5),
                                     constrained_layout=False)
                ax.set_position([0.15, 0.15, 0.7, 0.7])
            else:
                f = ax.figure

            setup_axes(ax)
            single_var_point_plot(data=data, x_var='BMI_b2', y_var=y_var, ax=ax, **kwargs)
            if y_var in self.plotting_labels:
                ax.set_ylabel(self.plotting_labels[y_var])
            ax.set_xlabel("BMI")
            xticks = np.arange(len(self.bmi_bin_centers2))
            ax.set_xticks(xticks)
            labels = self.bmi_bin_centers2[xticks]
            ax.set_xticklabels(labels, rotation=45)
            if yticks is None:
                fixed_yticks(ax, buffer=0.1, n_digits_input=n_digits)
            else:
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticks)
                if ylims is not None:
                    ax.set_ylim(ylims)
                else:
                    delta = (yticks[-1] - yticks[0])
                    ax.set_ylim(yticks[0] - buffer * delta, yticks[-1] + buffer * delta)

        return f, ax

    def plot_var_x_behav(self, data, y_var, behav_var, ax=None, yticks=None, n_digits=None, ylims=None, **kwargs):
        import matplotlib.pyplot as plt

        buffer = 0.1
        with plt.rc_context(rc=self.plotting_params):
            if ax is None:
                f, ax = plt.subplots(figsize=(self.age_axes_width,
                                              self.age_axes_width * 4 / 5),
                                     constrained_layout=False)
                ax.set_position([0.15, 0.15, 0.7, 0.7])
            else:
                f = ax.figure
            setup_axes(ax)

            single_var_point_plot(data=data, x_var=behav_var, y_var=y_var, ax=ax, **kwargs)
            if y_var in self.plotting_labels:
                ax.set_ylabel(self.plotting_labels[y_var])
            if behav_var in self.plotting_labels:
                ax.set_xlabel(self.plotting_labels[behav_var])

            if behav_var in 'sl_dur_mean_bin':
                labels = self.sl_dur_labels
            elif behav_var in 'sl_onset_mean_cat':
                labels = self.sl_onset_labels
            elif behav_var in 'wo_onset_cat':
                labels = self.wo_onset_labels
            elif behav_var in 'wo_dur_bin':
                labels = self.wo_dur_bin_labels
            elif behav_var in 'eTRIMP_bin':
                labels = self.eTRIMP_bin_labels
            elif behav_var in 'wo_norm_int_bin':
                labels = self.norm_intensity_labels
            elif behav_var in ['sl_dur_lvar_bin']:
                labels = []
                for ii, (k1, k2) in enumerate(zip(self.SL_DUR_LVAR_BINS[0:-1], self.SL_DUR_LVAR_BINS[1:])):
                    if ii == 0:
                        labels.append(f'[{k1:.2f}, {k2:.2f}]')
                    else:
                        labels.append(f'({k1:.2f}, {k2:.2f}]')

            xticks = np.arange(len(labels))
            ax.set_xticks(xticks)
            ax.set_xticklabels(labels, rotation=45)
            if yticks is None:
                fixed_yticks(ax, buffer=buffer, n_digits_input=n_digits)
            else:
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticks)
                if ylims is not None:
                    ax.set_ylim(ylims)
                else:
                    delta = (yticks[-1] - yticks[0])
                    ax.set_ylim(yticks[0] - buffer * delta, yticks[-1] + buffer * delta)

                ax.set_yticklabels(yticks)

    # ------------------------------------------------------------------
    # Bin definitions
    # ------------------------------------------------------------------
    def _age_bins(self, min_age=18, max_age=50, step=4):
        return self._get_bins(min_age, max_age, step)

    def _age_bins2(self, min_age=18, max_age=50, step=4):
        edges, _ = self._get_bins(min_age, max_age, step)
        centers = [f"({i},{j}]" for i, j in zip(edges[:-1], edges[1:])]
        centers[0] = f"[{edges[0]},{edges[1]}]"
        return edges, np.array(centers)

    def _bmi_bins(self, min_bmi=18, max_bmi=38, step=4):
        return self._get_bins(min_bmi, max_bmi, step)

    def _bmi_bins2(self, min_bmi=18, max_bmi=38, step=4):
        edges, _ = self._get_bins(min_bmi, max_bmi, step)
        edges = np.concatenate((np.array([0]), edges, np.array([100])))
        centers = [f"({i},{j}]" for i, j in zip(edges[:-1], edges[1:])]
        centers[0] = f'<={min_bmi}'
        centers[-1] = f'>{max_bmi}'
        return edges, np.array(centers)

    @staticmethod
    def _get_bins(min_val, max_val, step):
        edges = np.arange(min_val, max_val + 1, step)
        centers = (edges[:-1] + edges[1:]) / 2
        return edges, centers
