"""Public data-loading entry point.

`load_paper_data()` returns the three DataFrames every figure-family notebook
starts from:

  - `day_df`   one row per (n_id, date); the per-day biometric/sleep/workout time-series
  - `cycle_df` one row per (n_id, cycle_num); GEE-ready cycle-level table
  - `user_df`  one row per n_id; demographics, cycle counts, valid-user flag

On first call we parse the two CSVs (~30s for the 1.4M-row daily file), augment
with day-level workout columns, build the cycle and user tables, and cache
parquet copies under `data/cache/`. Subsequent calls read from cache.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .behaviors import (
    add_workout_day_columns,
    aggregate_sleep_per_cycle,
    aggregate_workouts_per_cycle,
)
from .cycles import build_cycle_table, build_user_table


_DAY_PARQUET = "day.parquet"
_CYCLE_PARQUET = "cycle.parquet"
_USER_PARQUET = "user.parquet"


def load_paper_data(
    *,
    data_dir: Path | str | None = None,
    use_cache: bool = True,
    rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns `(day_df, cycle_df, user_df)`.

    Parameters
    ----------
    data_dir : path, optional
        Override the default `data/` directory.
    use_cache : bool, default True
        If True, read from parquet caches under `data_dir/cache/` when present.
    rebuild : bool, default False
        Force a rebuild even if caches exist (re-parses CSVs and re-aggregates).
    """
    data_dir = Path(data_dir) if data_dir is not None else config.DATA_DIR
    cache_dir = data_dir / "cache"

    if use_cache and not rebuild and _all_caches_present(cache_dir):
        day_df = pd.read_parquet(cache_dir / _DAY_PARQUET)
        cycle_df = pd.read_parquet(cache_dir / _CYCLE_PARQUET)
        user_df = pd.read_parquet(cache_dir / _USER_PARQUET)
        return day_df, cycle_df, user_df

    day_df, summary_df = _load_csvs(data_dir)
    day_df = add_workout_day_columns(day_df)

    user_df = build_user_table(day_df, summary_df)
    cycle_df = build_cycle_table(day_df, user_df)
    cycle_df = aggregate_sleep_per_cycle(day_df, cycle_df)
    cycle_df = aggregate_workouts_per_cycle(day_df, cycle_df)

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _to_parquet_safe(day_df, cache_dir / _DAY_PARQUET)
        _to_parquet_safe(cycle_df.reset_index(), cache_dir / _CYCLE_PARQUET)
        _to_parquet_safe(user_df.reset_index(), cache_dir / _USER_PARQUET)

    return day_df, cycle_df, user_df


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _all_caches_present(cache_dir: Path) -> bool:
    return all((cache_dir / f).exists() for f in (_DAY_PARQUET, _CYCLE_PARQUET, _USER_PARQUET))


def _to_parquet_safe(df: pd.DataFrame, path: Path) -> None:
    """`pd.cut` produces dictionary-encoded Interval columns that pyarrow can't
    write to parquet. Cast those to plain string before persisting; the
    categorical structure can be rebuilt downstream from `config` if needed.
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_categorical_dtype(s) and pd.api.types.is_interval_dtype(s.cat.categories):
            out[col] = s.astype(str)
    out.to_parquet(path)


def _load_csvs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_csv = data_dir / config.DAILY_CSV.name
    summary_csv = data_dir / config.SUMMARY_CSV.name

    if not daily_csv.exists() or not summary_csv.exists():
        raise FileNotFoundError(
            f"Expected CSV files at {daily_csv} and {summary_csv}.\n"
            "These are private and gitignored. See README.md for data-access "
            "instructions (per the WHOOP data agreement)."
        )

    day_df = pd.read_csv(daily_csv, index_col=0)
    summary_df = pd.read_csv(summary_csv, index_col=0)
    return day_df, summary_df
