"""Reproduction codebase for "The menstrual cycle through the lens of a wearable
device" (npj Digital Medicine).

Public API:
  - `load_paper_data()`             : entry point, returns (day_df, cycle_df, user_df)
  - `build_user_table`, `build_cycle_table`, `add_delta_thresholds` : cycle/user tables
  - `aggregate_sleep_per_cycle`, `aggregate_workouts_per_cycle`     : behaviors
  - `fit_gee`, `coef_table`         : statsmodels GEE wrapper
  - `circmean_day`, `circvar_day`, `mad` : circular statistics helpers
  - `PLOTTING_ROUTINES`             : matplotlib styling + bootstrap-CI plots
"""

__version__ = "0.1.0"
__author__ = "Alex Gonzalez"

from .behaviors import (
    add_workout_day_columns,
    aggregate_sleep_per_cycle,
    aggregate_workouts_per_cycle,
)
from .cycles import (
    add_delta_thresholds,
    build_cycle_table,
    build_user_table,
    compute_bin_weights,
)
from .io import load_paper_data
from .plotting_routines import PLOTTING_ROUTINES
from .stats import circmean_day, circvar_day, coef_table, fit_gee, mad

__all__ = [
    "load_paper_data",
    "build_user_table",
    "build_cycle_table",
    "add_delta_thresholds",
    "compute_bin_weights",
    "add_workout_day_columns",
    "aggregate_sleep_per_cycle",
    "aggregate_workouts_per_cycle",
    "fit_gee",
    "coef_table",
    "circmean_day",
    "circvar_day",
    "mad",
    "PLOTTING_ROUTINES",
]
