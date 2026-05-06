"""Thin wrapper around `statsmodels.formula.api.gee`.

Mirrors the call pattern in
`whoop_analyses/whoop_analyses/paper_code_wrapper.py:445-468`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


_COV_STRUCTS = {
    "exchangeable": sm.cov_struct.Exchangeable,
    "independence": sm.cov_struct.Independence,
    "ar": sm.cov_struct.Autoregressive,
}

_FAMILIES = {
    "gaussian": sm.families.Gaussian,
    "binomial": sm.families.Binomial,
}


def fit_gee(
    formula: str,
    data: pd.DataFrame,
    *,
    groups: str = "n_id",
    family: str = "gaussian",
    cov_struct: str = "exchangeable",
    weights: str | None = None,
    drop_na: bool = True,
):
    """Fit a GEE.

    Parameters
    ----------
    formula : str
        R-style formula, e.g. `'cycle_length ~ age + age2 + cos_season'`.
    data : DataFrame
        Must contain `groups` and any column referenced by `formula` and `weights`.
    groups : str
        Cluster variable for the GEE. Defaults to `'n_id'` (subject).
    family : {'gaussian', 'binomial'}
    cov_struct : {'exchangeable', 'independence', 'ar'}
    weights : str | None
        Column name in `data` to use as observation weights.
    drop_na : bool
        Drop rows missing any value used in the formula or weights before fitting.
    """
    if drop_na:
        used = _formula_columns(formula) + ([weights] if weights else []) + [groups]
        data = data.dropna(subset=[c for c in used if c in data.columns]).copy()

    fit_kwargs = dict(
        formula=formula,
        groups=data[groups],
        data=data,
        family=_FAMILIES[family](),
        cov_struct=_COV_STRUCTS[cov_struct](),
    )
    if weights is not None:
        fit_kwargs["weights"] = data[weights]

    return smf.gee(**fit_kwargs).fit()


def coef_table(result, *, expo: bool = False, alpha: float = 0.05) -> pd.DataFrame:
    """Tidy coefficient table: `est, SE, z, p, CI_lo, CI_hi` (and `OR, OR_lo, OR_hi`
    if `expo=True`, useful for binomial fits).
    """
    params = result.params
    se = result.bse
    z = result.tvalues
    p = result.pvalues
    ci = result.conf_int(alpha=alpha)

    out = pd.DataFrame({
        "est": params,
        "SE": se,
        "z": z,
        "p": p,
        "CI_lo": ci[0],
        "CI_hi": ci[1],
    })
    if expo:
        out["OR"] = np.exp(out["est"])
        out["OR_lo"] = np.exp(out["CI_lo"])
        out["OR_hi"] = np.exp(out["CI_hi"])
    return out


def _formula_columns(formula: str) -> list[str]:
    """Best-effort: extract column names referenced by an R-style formula."""
    import re

    rhs = formula.split("~", 1)[-1]
    lhs = formula.split("~", 1)[0].strip()
    tokens = re.split(r"[\s+\-*:/()]+", rhs)
    return [t for t in [lhs, *tokens] if t and not t.isdigit()]
