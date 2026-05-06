"""Circular statistics on hour-of-day variables (sleep/workout onset times).

Ported verbatim from `whoop_analyses/whoop_analyses/utils.py:150-197`.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def mad(x):
    """Median absolute deviation, NaN-omitting."""
    return stats.median_abs_deviation(x, nan_policy="omit")


def circmean_day(x):
    """Circular mean for hour-of-day values, returned in hours [0, 24)."""
    return stats.circmean(2 * np.pi * x / 24, nan_policy="omit") / (2 * np.pi) * 24


def circvar_day(x):
    """Circular variance for hour-of-day values (dimensionless, in [0, 1])."""
    return stats.circvar(2 * np.pi * x / 24, nan_policy="omit")
