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
    """
    data_dir = Path(data_dir) if data_dir is not None else config.DATA_DIR
    day_df, summary_df = _load_csvs(data_dir)
    CBM = CycleBehavMethods(day_df, summary_df)
    return day_df, CBM


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
