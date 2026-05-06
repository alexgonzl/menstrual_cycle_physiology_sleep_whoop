"""User and cycle table construction.

The provided daily CSV is already cycle-aligned (`cycle_num`, `cycle_days`,
`starts`, `flags`, `n_prev_valid`). Here we aggregate it into a per-user
table (one row per user) and a per-cycle table (one row per (n_id, cycle_num)),
applying the paper's validity rule:

- A cycle is *valid* iff every day in the cycle has `flags == 0`. This matches
  the original (`whoop_analyses/whoop_analyses/src.py:1047`):
  `vcl = cl[cycle_groups["flags"].mean() == 0]`. Boundary cycles
  (`flags == -1`) and anomalous cycles (`flags == 1`) are dropped wholesale.
- Valid users have median cycle length in [21, 35] and >=5 valid cycles.

The previous `CYCLE_LENGTH_ANALYSIS` class accomplished the same thing but in
a stateful, mutating-self style with several bugs. We rewrite as pure functions
returning new DataFrames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .stats.circular import mad


# ---------------------------------------------------------------------------
# user table
# ---------------------------------------------------------------------------
def build_user_table(day_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """One row per user. Joins demographics from `summary_df` and computes
    cycle counts/means/medians from `day_df`.

    Returns
    -------
    DataFrame indexed by `n_id` with columns
    `n_c, n_vc, length, mean_cl, sd_cl, median_cl, mad_cl, valid_user, E_n_c,
    age, age2, BMI, BMI2, age_b, BMI_b`.
    """
    cycle_groups = day_df.groupby(["n_id", "cycle_num"], observed=True)

    cl_per_cycle = cycle_groups.size()  # cycle length = day count per cycle
    # A cycle is valid iff every day in it has flags == 0. Mirrors src.py:1047.
    valid_cycle = cycle_groups["flags"].mean() == 0

    user = pd.DataFrame(index=day_df["n_id"].unique())
    user.index.name = "n_id"

    user["n_c"] = cl_per_cycle.groupby("n_id").count()
    user["n_vc"] = cl_per_cycle[valid_cycle].groupby("n_id").count()
    user["length"] = day_df.groupby("n_id")["date"].size()

    user["mean_cl"] = cl_per_cycle[valid_cycle].groupby("n_id").mean()
    user["sd_cl"] = cl_per_cycle[valid_cycle].groupby("n_id").std()
    user["median_cl"] = cl_per_cycle[valid_cycle].groupby("n_id").median()
    user["mad_cl"] = cl_per_cycle[valid_cycle].groupby("n_id").apply(mad)

    user["valid_user"] = (
        (user["median_cl"] >= config.MIN_MEDIAN_CL)
        & (user["median_cl"] <= config.MAX_MEDIAN_CL)
        & (user["n_vc"] >= config.MIN_N_VCL)
    )
    user["E_n_c"] = np.round(user["length"] / user["mean_cl"])

    user = user[user["valid_user"]].copy()

    for col in ["age", "BMI"]:
        user[col] = user.index.map(summary_df[col])
        user[f"{col}2"] = user[col] ** 2

    age_edges, age_centers = _step_bins(config.MIN_AGE, config.MAX_AGE, config.AGE_BIN_STEP)
    bmi_edges, bmi_centers = _step_bins(config.MIN_BMI, config.MAX_BMI, config.BMI_BIN_STEP)
    user["age_b"] = pd.cut(user["age"], age_edges, labels=age_centers.astype(int), include_lowest=True)
    user["BMI_b"] = pd.cut(user["BMI"], bmi_edges, labels=bmi_centers.astype(int), include_lowest=True)

    return user


# ---------------------------------------------------------------------------
# cycle table
# ---------------------------------------------------------------------------
def build_cycle_table(day_df: pd.DataFrame, user_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (n_id, cycle_num) for users in `user_df`. Adds cycle length,
    deviation from per-user median, age/BMI joined from the user table, and
    seasonality (cosine/sine of cycle-onset day-of-year).

    Filters to:
      - users present in `user_df` (i.e. valid users)
      - cycle length in [MIN_CL, MAX_CL]
      - bounded cycles only (`flags == 0`); drops the head/tail unbounded ones
    """
    df = day_df[day_df["n_id"].isin(user_df.index)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.dayofyear

    cycle_groups = df.groupby(["n_id", "cycle_num"], observed=True)

    cl = cycle_groups.size()
    # A cycle is valid iff every day in it has flags == 0 (matches src.py:1047).
    valid_cycle = cycle_groups["flags"].mean() == 0

    cycle = pd.DataFrame({"length": cl})
    cycle["vcl"] = valid_cycle

    for metric in ["median_cl", "mean_cl", "mad_cl", "sd_cl", "age", "age2", "BMI", "BMI2", "age_b", "BMI_b"]:
        cycle[metric] = cycle.index.get_level_values("n_id").map(user_df[metric])

    cycle["delta_cl"] = cycle["length"] - cycle["median_cl"]
    cycle["z_delta_cl"] = (cycle["length"] - cycle["mean_cl"]) / cycle["sd_cl"]

    cycle = cycle[cycle["vcl"]].copy()

    cycle["data_day"] = cycle_groups["day"].first()
    cycle["day_of_year"] = cycle_groups["day_of_year"].first()
    cycle["cycle_length"] = cycle["length"]
    cycle["cos_season"] = np.cos(2 * np.pi * cycle["day_of_year"] / 365.25)
    cycle["sin_season"] = np.sin(2 * np.pi * cycle["day_of_year"] / 365.25)

    return cycle


def add_delta_thresholds(
    cycle_df: pd.DataFrame,
    user_df: pd.DataFrame,
    *,
    max_dev: int = config.MAX_DAY_DEV,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Adds `a_delta_cl_ge_{thr}` columns to `cycle_df` and the same per-user
    proportions to `user_df`. Returns (cycle_df, user_df, list_of_added_col_names)
    so callers don't have to know the threshold range.
    """
    cycle_df = cycle_df.copy()
    user_df = user_df.copy()
    cols: list[str] = []
    for thr in range(0, max_dev):
        col = f"a_delta_cl_ge_{thr}"
        y = (cycle_df["delta_cl"].abs() >= thr).astype(int)
        cycle_df[col] = y
        user_df[col] = y.groupby(level="n_id").mean()
        cols.append(col)
    return cycle_df, user_df, cols


# ---------------------------------------------------------------------------
# bin weights for unequal subgroup sizes
# ---------------------------------------------------------------------------
def compute_bin_weights(
    table: pd.DataFrame,
    col: str,
    *,
    raw_col: str | None = None,
    edges: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Add a `<col>_weights` column = 1/log(N_in_bin), with edge-bin re-mapping
    for values that fall below/above the bin range.

    Implements the weighting scheme used in the paper's GEE models. See
    `cycle_length_analyses.py:445-499` in the original code.
    """
    table = table.copy()
    weights_col = f"{col}_weights"

    if raw_col is None:
        for suffix in ("_bin", "_b2", "_b", "_cat_num", "_cat"):
            if col.endswith(suffix):
                raw_col = col[: -len(suffix)]
                break
        else:
            raw_col = col

    table[weights_col] = (
        table.groupby(col, observed=True)[col]
        .transform(lambda x: 1 / np.log(len(x)) if len(x) > 0 else 0)
    )

    cats = table[col].cat.categories if hasattr(table[col], "cat") else None
    if cats is not None and len(cats) > 0 and edges is not None:
        lo, hi = edges
        low_mask = table[col] == cats[0]
        high_mask = table[col] == cats[-1]
        if low_mask.any() and raw_col in table.columns:
            w_low = table.loc[low_mask, weights_col].iloc[0]
            table.loc[table[raw_col] < lo, weights_col] = w_low
        if high_mask.any() and raw_col in table.columns:
            w_high = table.loc[high_mask, weights_col].iloc[0]
            table.loc[table[raw_col] > hi, weights_col] = w_high

    return table


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _step_bins(min_val: float, max_val: float, step: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(min_val, max_val + 1, step)
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers
