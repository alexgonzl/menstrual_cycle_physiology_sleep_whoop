"""Per-cycle aggregation of sleep and workout behaviors.

Ported from `whoop_analyses/whoop_analyses/cl_behav_methods.py`:
  - day-level workout columns: lines 224-244 (`_add_workout_categories`)
  - sleep cycle aggregation:    lines 557-626 (`add_sleep_behaviors`)
  - workout cycle aggregation:  lines 628-670 (`add_workout_behaviors`)

The provided daily CSV does not include the day-level helper columns the
original computed (`n_wo`, `total_wo_duration`, `total_intensity_eTRIMP`,
`wo_time_cos/sin`, `n_morning_wo`, `n_night_wo`); we derive them in
`add_workout_day_columns` before cycle-level aggregation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .cycles import compute_bin_weights
from .stats.circular import circmean_day, circvar_day


# ---------------------------------------------------------------------------
# day-level workout helpers (mirror cl_behav_methods.py:224-244)
# ---------------------------------------------------------------------------
def add_workout_day_columns(day_df: pd.DataFrame) -> pd.DataFrame:
    """Add the day-level workout aggregates the cycle-level aggregator expects.

    Computes from the per-workout columns (`activity_{0,1,2}`,
    `wo_time_{0,1,2}`, `intensity_{0,1,2}`, `duration_{0,1,2}`):
      - `wo_time_cos`, `wo_time_sin` (mean across the day's workouts)
      - `n_morning_wo`, `n_night_wo`, `n_wo`
      - `total_wo_duration` (minutes)
      - `total_intensity_eTRIMP` (sum of HR-zone time × zone weight)
    """
    df = day_df.copy()

    # Mask out low-intensity workouts (treated as missing).
    intensity_cols = [f"intensity_{i}" for i in range(3)]
    wo_time_cols = [f"wo_time_{i}" for i in range(3)]
    for i, wt_col in enumerate(wo_time_cols):
        low = df[intensity_cols[i]] < config.MIN_WO_INTENSITY
        df.loc[low, wt_col] = np.nan

    # Bin workout time-of-day into morning/night for `n_morning_wo` / `n_night_wo`.
    bin_edges = [0, 5, 8, 18, 24]
    bin_labels = ["0-5", "5-8", "8-18", "18-24"]
    binned = pd.DataFrame({
        c: pd.cut(df[c], bin_edges, labels=bin_labels, include_lowest=True)
        for c in wo_time_cols
    })

    df["wo_time_cos"] = np.cos(df[wo_time_cols] * 2 * np.pi / 24).mean(axis=1)
    df["wo_time_sin"] = np.sin(df[wo_time_cols] * 2 * np.pi / 24).mean(axis=1)
    df["n_morning_wo"] = (binned == "5-8").sum(axis=1)
    df["n_night_wo"] = (binned == "18-24").sum(axis=1)
    df["n_wo"] = (~binned.isna()).sum(axis=1)

    duration_cols = [f"duration_{i}" for i in range(3)]
    df["total_wo_duration"] = df[duration_cols].sum(axis=1) * 60  # hours -> minutes

    # eTRIMP = sum_i (i * z_i)  where z_i is time spent in HR zone i.
    df["total_intensity_eTRIMP"] = (df[config.HR_ZONES] * config.HR_WEIGHTS).sum(axis=1)

    return df


# ---------------------------------------------------------------------------
# cycle-level aggregation
# ---------------------------------------------------------------------------
def aggregate_sleep_per_cycle(day_df: pd.DataFrame, cycle_df: pd.DataFrame) -> pd.DataFrame:
    """Add sleep behavior aggregates to `cycle_df`. Returns a new DataFrame.

    Adds: `sl_onset_mean/cos/sin/var`, `sl_dur_mean/var/sd/lvar/lvar2/mean2`,
    `sl_TiB_mean/var/lvar/lvar2/mean2`, plus categorical bins
    (`sl_onset_mean_cat`, `sl_dur_mean_cat/_bin`, `sl_dur_var_bin`,
    `sl_dur_lvar_bin`, `sl_dur_sd_bin`, `sl_TiB_mean_bin`) and their
    `*_weights` columns.
    """
    cycle_df = cycle_df.copy()
    grouped = day_df.groupby(["n_id", "cycle_num"], observed=True)

    sl_onset_mean = grouped["sl_onset"].apply(circmean_day)
    cycle_df["sl_onset_mean"] = sl_onset_mean
    cycle_df["sl_onset_cos"] = np.cos(2 * np.pi * sl_onset_mean / 24)
    cycle_df["sl_onset_sin"] = np.sin(2 * np.pi * sl_onset_mean / 24)
    cycle_df["sl_onset_var"] = grouped["sl_onset"].apply(circvar_day)

    sl_dur_mean = grouped["sleep_dur"].mean()
    sl_dur_var = grouped["sleep_dur"].var()
    sl_dur_sd = grouped["sleep_dur"].std()
    cycle_df["sl_dur_mean"] = sl_dur_mean
    cycle_df["sl_dur_var"] = sl_dur_var
    cycle_df["sl_dur_sd"] = sl_dur_sd
    cycle_df["sl_dur_lvar"] = np.log2(sl_dur_var)
    cycle_df["sl_dur_lvar2"] = cycle_df["sl_dur_lvar"] ** 2
    cycle_df["sl_dur_mean2"] = sl_dur_mean ** 2

    sl_TiB_mean = grouped["time_in_bed"].mean()
    sl_TiB_var = grouped["time_in_bed"].var()
    cycle_df["sl_TiB_mean"] = sl_TiB_mean
    cycle_df["sl_TiB_var"] = sl_TiB_var
    cycle_df["sl_TiB_lvar"] = np.log2(sl_TiB_var)
    cycle_df["sl_TiB_lvar2"] = cycle_df["sl_TiB_lvar"] ** 2
    cycle_df["sl_TiB_mean2"] = sl_TiB_mean ** 2

    binning_ops = {
        "sl_onset_mean_cat": (sl_onset_mean, config.SL_ONSET_BIN_EDGES, config.SL_ONSET_BIN_LABELS),
        "sl_dur_mean_cat":   (sl_dur_mean,   config.SL_DUR_CAT_EDGES,   config.SL_DUR_CAT_LABELS),
        "sl_dur_mean_bin":   (sl_dur_mean,   config.SL_DUR_BIN_EDGES,   config.SL_DUR_BIN_LABELS),
        "sl_dur_var_bin":    (sl_dur_var,    config.SL_DUR_VAR_BINS,    None),
        "sl_dur_lvar_bin":   (cycle_df["sl_dur_lvar"], config.SL_DUR_LVAR_BINS, None),
        "sl_dur_sd_bin":     (sl_dur_sd,     config.SL_DUR_SD_BINS,     None),
        "sl_TiB_mean_bin":   (sl_TiB_mean,   config.SL_DUR_BIN_EDGES,   config.SL_DUR_BIN_LABELS),
    }
    for name, (data_col, bins, labels) in binning_ops.items():
        if labels is not None:
            cycle_df[name] = pd.cut(data_col, bins, labels=labels, include_lowest=True)
        else:
            cycle_df[name] = pd.cut(data_col, bins, include_lowest=True)
        cycle_df = compute_bin_weights(cycle_df, name)

    return cycle_df


def aggregate_workouts_per_cycle(day_df: pd.DataFrame, cycle_df: pd.DataFrame) -> pd.DataFrame:
    """Add workout behavior aggregates to `cycle_df`. Returns a new DataFrame.

    Day-level inputs (`n_wo`, `total_wo_duration`, `total_intensity_eTRIMP`,
    `wo_time_cos/sin`, `n_morning_wo`, `n_night_wo`) are computed by
    `add_workout_day_columns` if missing.
    """
    if "total_intensity_eTRIMP" not in day_df.columns:
        day_df = add_workout_day_columns(day_df)

    cycle_df = cycle_df.copy()
    grouped = day_df.groupby(["n_id", "cycle_num"], observed=True)

    cycle_df["n_morning_wo"] = grouped["n_morning_wo"].sum()
    cycle_df["n_night_wo"] = grouped["n_night_wo"].sum()
    cycle_df["n_wo"] = grouped["n_wo"].sum()
    cycle_df["wo_time_index"] = (
        (cycle_df["n_morning_wo"] - cycle_df["n_night_wo"])
        / (cycle_df["n_morning_wo"] + cycle_df["n_night_wo"])
    )
    cycle_df["wo_rate"] = cycle_df["n_wo"] / cycle_df["length"]
    cycle_df["wo_time_cos"] = grouped["wo_time_cos"].mean()
    cycle_df["wo_time_sin"] = grouped["wo_time_sin"].mean()

    cycle_df["total_wo_duration"] = grouped["total_wo_duration"].mean()
    cycle_df["total_intensity_eTRIMP"] = grouped["total_intensity_eTRIMP"].mean()
    cycle_df["eTRIMP"] = cycle_df["total_intensity_eTRIMP"]
    cycle_df["eTRIMP2"] = cycle_df["eTRIMP"] ** 2
    cycle_df["norm_intensity"] = cycle_df["eTRIMP"] / cycle_df["total_wo_duration"]
    cycle_df["norm_intensity2"] = cycle_df["norm_intensity"] ** 2

    cycle_df["eTRIMP_bin"] = pd.cut(
        cycle_df["eTRIMP"], bins=config.eTRIMP_BIN_EDGES,
        labels=config.eTRIMP_BIN_LABELS, include_lowest=True,
    )
    cycle_df["wo_dur_bin"] = pd.cut(
        cycle_df["total_wo_duration"], config.WO_DUR_BIN_EDGES,
        labels=config.WO_DUR_BIN_LABELS, include_lowest=True,
    )
    cycle_df["wo_norm_int_bin"] = pd.cut(
        cycle_df["norm_intensity"], config.WO_NORM_INT_BIN_EDGES,
        labels=config.WO_NORM_INT_BIN_LABELS, include_lowest=True,
    )

    cycle_df = compute_bin_weights(cycle_df, "eTRIMP_bin")
    return cycle_df
