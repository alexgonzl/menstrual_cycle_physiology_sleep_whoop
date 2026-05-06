"""Constants and bin definitions used across analyses.

Lifted from the original `cycle_length_analyses.py`. One source of truth so all
notebooks see identical thresholds and bin edges.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ---- paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

DAILY_CSV = DATA_DIR / "physio_sleep_mc_power_users_daily_data.csv"
SUMMARY_CSV = DATA_DIR / "power_users_summary_table.csv"

# ---- cycle length thresholds ----------------------------------------------
MIN_CL = 15
MAX_CL = 45
MIN_MEDIAN_CL = 21
MAX_MEDIAN_CL = 35
MIN_N_VCL = 5
PHASE_LENGTH_DAYS = 7
MAX_DAY_DEV = 10  # range over which a_delta_cl_ge_{thr} columns are emitted

# ---- demographics ----------------------------------------------------------
MIN_AGE = 18
MAX_AGE = 50
AGE_BIN_STEP = 4

MIN_BMI = 18
MAX_BMI = 38
BMI_BIN_STEP = 4

# ---- sleep duration --------------------------------------------------------
SL_DUR_CAT_EDGES = [4, 6.5, 8.5, 11]
SL_DUR_CAT_LABELS = ["low", "medium", "high"]

SL_DUR_BIN_EDGES = [4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
SL_DUR_BIN_LABELS = [5, 6, 7, 8, 9]

# sleep duration variability bins (log2-spaced)
SL_DUR_LVAR_BINS = np.array([-4, -2, 0, 2, 4], dtype=float)
SL_DUR_VAR_BINS = 2 ** SL_DUR_LVAR_BINS
SL_DUR_SD_BINS = np.sqrt(SL_DUR_VAR_BINS)

# ---- sleep onset -----------------------------------------------------------
SL_ONSET_BIN_EDGES = [0, 6, 20, 22, 24]
SL_ONSET_BIN_LABELS = ["post_midnight", "day", "evening", "night"]

# ---- workouts --------------------------------------------------------------
MIN_WO_INTENSITY = 30  # below this the workout is treated as missing

eTRIMP_BIN_EDGES = np.array([0, 60, 120, 180, 240, 300, 600])
eTRIMP_BIN_LABELS = ["very_low", "low", "medium", "high", "very_high", "extreme"]

WO_DUR_BIN_EDGES = np.array([0, 30, 60, 90, 120, 150, 180, 210, 500])
WO_DUR_BIN_LABELS = ["0-30", "30-60", "60-90", "90-120", "120-150", "150-180", "180-210", "210+"]

WO_NORM_INT_BIN_EDGES = np.array([0.5, 2.25, 3.5, 5.5])
WO_NORM_INT_BIN_LABELS = ["low", "medium", "high"]

WO_ONSET_BIN_EDGES = [0, 4, 6, 10, 16, 20, 22, 24]
WO_ONSET_BIN_LABELS = ["0-4am", "4am-6pm", "6-10am", "10am-4pm", "4-8pm", "8-10pm", "10am-12pm"]

# HR zones — eTRIMP weights are the zone index (z0 → 0, z1 → 1, …, z5 → 5)
HR_ZONES = [f"z{i}" for i in range(0, 6)]
HR_WEIGHTS = np.arange(6)

# ---- biometrics ------------------------------------------------------------
BIOMETRICS = ["RHR", "HRV", "RR", "skin_temp", "blood_oxygen"]

# ---- figure styling --------------------------------------------------------
FIGURE_DPI = 500
HEATMAP_FORMATS = ("png",)         # heatmaps: PNG only (SVG too large)
DEFAULT_FIGURE_FORMATS = ("svg", "png")
