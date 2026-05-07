"""Reproduction codebase for "The menstrual cycle through the lens of a wearable
device" (npj Digital Medicine).

Public API:
  - `load_paper_data()`      : entry point, returns (day_df, CBM)
  - `CycleBehavMethods`      : ported from cl_behav_methods.py
  - `CycleLengthAnalyses`    : ported from paper_code_wrapper.py
  - `StatisticalPredictionHandler` : ported from statistical_prediction_methods.py
  - `circmean_day`, `circvar_day`, `mad`
  - `PLOTTING_ROUTINES`
"""

__version__ = "0.1.0"
__author__ = "Alex Gonzalez"

from . import config
from .cl_behav_methods import CycleBehavMethods
from .io import load_paper_data
from .paper_code_wrapper import CycleLengthAnalyses
from .pca_analyses import Biometrics_VAR
from .physio_behavior_analyses import PhysioBehaviorAnalyses
from .physio_methods import PhysioBehavChangeMethods, PhysioMethods
from .plotting_routines import PLOTTING_ROUTINES
from .stats import circmean_day, circvar_day, mad
from .stats.contrasts import StatisticalPredictionHandler

__all__ = [
    "load_paper_data",
    "CycleBehavMethods",
    "CycleLengthAnalyses",
    "PhysioMethods",
    "PhysioBehavChangeMethods",
    "PhysioBehaviorAnalyses",
    "Biometrics_VAR",
    "StatisticalPredictionHandler",
    "circmean_day",
    "circvar_day",
    "mad",
    "PLOTTING_ROUTINES",
    "config",
]
