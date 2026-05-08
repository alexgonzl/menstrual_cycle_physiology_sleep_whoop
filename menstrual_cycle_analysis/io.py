"""Public data-loading entry point.

`load_paper_data()` reads the two private CSVs, instantiates `CycleBehavMethods`
(which runs the full preprocessing pipeline in its constructor — derives
j_cycle columns, drops unbounded cycles, builds user and cycle tables), and
returns `(day_df, CBM)`. Notebooks then call `CBM.add_sleep_behaviors(...)`,
`CBM.add_workout_behaviors(...)` and instantiate
`CycleLengthAnalyses(CBM=CBM)` for plotting.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .cl_behav_methods import CycleBehavMethods


def load_paper_data(
    *,
    data_dir: Path | str | None = None,
    daily_csv: Path | str | None = None,
    summary_csv: Path | str | None = None,
) -> tuple[pd.DataFrame, CycleBehavMethods]:
    """Returns `(day_df, CBM)`.

    `day_df` is the raw daily CSV (1.4M rows). `CBM` is an initialised
    `CycleBehavMethods` instance whose `tables['user']` and `tables['cycle']`
    are populated; the cycle aggregation steps (sleep, workouts) are
    invoked by the caller per the source notebook flow:

        CBM.add_sleep_behaviors('user')
        CBM.add_sleep_behaviors('cycle')
        CBM.add_workout_behaviors('user')
        CBM.add_workout_behaviors('cycle')

    Parameters
    ----------
    data_dir : optional
        Directory containing the two CSVs. Defaults to `config.DATA_DIR`.
        Within this directory the loader looks for the filenames at
        `config.DAILY_CSV.name` and `config.SUMMARY_CSV.name`.
    daily_csv, summary_csv : optional
        Explicit paths to the daily and per-participant CSVs, overriding
        `data_dir` + the configured filenames. Use these to load CSVs that
        are named differently, or that live in different directories.
    """
    daily_path, summary_path = _resolve_csv_paths(data_dir, daily_csv, summary_csv)
    day_df, summary_df = _load_csvs(daily_path, summary_path)
    CBM = CycleBehavMethods(day_df, summary_df)
    return day_df, CBM


def _resolve_csv_paths(
    data_dir: Path | str | None,
    daily_csv: Path | str | None,
    summary_csv: Path | str | None,
) -> tuple[Path, Path]:
    base = Path(data_dir) if data_dir is not None else config.DATA_DIR
    daily = Path(daily_csv) if daily_csv is not None else base / config.DAILY_CSV.name
    summary = Path(summary_csv) if summary_csv is not None else base / config.SUMMARY_CSV.name
    return daily, summary


def _load_csvs(daily_csv: Path, summary_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not daily_csv.exists() or not summary_csv.exists():
        raise FileNotFoundError(
            f"Expected CSV files at {daily_csv} and {summary_csv}.\n"
            "These are private and gitignored. See README.md for data-access "
            "instructions (per the WHOOP data agreement). To point at CSVs with "
            "different names or paths, pass daily_csv= and summary_csv= to "
            "load_paper_data()."
        )

    # Daily CSV is the slow one (~25s for 1.3M rows). Cache the parsed
    # DataFrame as parquet on first parse; subsequent calls read parquet.
    # No mtime/version checks — invalidation is manual: `rm -rf data/cache/`.
    cache_path = config.CACHE_DIR / "day_df.parquet"
    if cache_path.exists():
        day_df = pd.read_parquet(cache_path)
    else:
        day_df = pd.read_csv(daily_csv, index_col=0)
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        day_df.to_parquet(cache_path)

    summary_df = pd.read_csv(summary_csv, index_col=0)
    return day_df, summary_df
