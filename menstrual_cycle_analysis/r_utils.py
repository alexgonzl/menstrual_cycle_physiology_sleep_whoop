"""
Utilities to save and load R objects (e.g. GAM fits) from Python using rpy2.
This module provides two main functions:

- save_r_object(obj, path): save an R object to disk using saveRDS
- load_r_object(path): load an R object from disk using readRDS

It also provides thin helpers around dictionaries of models (e.g. pm.gam_models)
so the existing code can call `save_gam_models(pm, path)` and `load_gam_models(pm, path)`.

Notes:
- This file uses rpy2. If rpy2 is not installed, functions will raise an ImportError with
  a helpful message explaining how to install it.
- The I/O uses R's native saveRDS/readRDS which preserves R object structures.

"""

from pathlib import Path
from typing import Dict


def _require_rpy2():
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import r
        from rpy2.robjects import globalenv
        from rpy2.robjects.packages import importr
        return ro, r, globalenv, importr
    except Exception as e:
        raise ImportError(
            "rpy2 is required to use r_utils. Install it with `pip install rpy2`and ensure a compatible R installation is available on your PATH."
        ) from e


def save_r_object(r_object, path: str):
    """Save an R object to disk using R's saveRDS.

    Parameters
    ----------
    r_object : rpy2.robjects.RObject
        An R object created/held via rpy2 (for example a fitted mgcv::gam object).
    path : str
        Destination filename. If extension is not '.rds', it will be appended.
    """
    ro, r, _, _ = _require_rpy2()
    p = Path(path)
    if p.suffix != '.rds':
        p = p.with_suffix('.rds')
    r['saveRDS'](r_object, str(p))
    return str(p)


def load_r_object(path: str):
    """Load an R object from disk using R's readRDS and return the rpy2 R object.

    Parameters
    ----------
    path : str
        Path to .rds file

    Returns
    -------
    rpy2.robjects.RObject
    """
    ro, r, _, _ = _require_rpy2()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"RDS file not found: {p}")
    return r['readRDS'](str(p))


# Helpers for pm.gam_models which is expected to be a dict-like structure
# mapping keys (biometric names) to R gam objects (held via rpy2) or to native python wrappers.

def save_gam_models(pm, directory: str, overwrite: bool = False) -> Dict[str, str]:
    """Save pm.gam_models to the specified directory as individual .rds files.

    Returns a dict mapping model keys to saved file paths.
    """
    ro, r, _, _ = _require_rpy2()
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    saved = {}
    if not hasattr(pm, 'gam_models'):
        raise AttributeError('pm object must have attribute gam_models')
    for key, model in pm.gam_models.items():
        filename = d / f"{key}.rds"
        if filename.exists() and not overwrite:
            saved[key] = str(filename)
            continue
        # model is expected to already be an rpy2 object; if it's a wrapper with an .r attribute, use that
        r_obj = getattr(model, 'r', model)
        r['saveRDS'](r_obj, str(filename))
        saved[key] = str(filename)
    return saved


def load_gam_models(pm, directory: str, keys=None) -> Dict[str, object]:
    """Load GAM models from a directory into pm.gam_models. Returns the dict of loaded models.

    If keys is provided, only those filenames (without .rds) will be loaded.
    """
    ro, r, _, _ = _require_rpy2()
    d = Path(directory)
    if not d.exists():
        raise FileNotFoundError(f"Directory not found: {d}")
    files = list(d.glob('*.rds'))
    loaded = {}
    for f in files:
        key = f.stem
        if keys is not None and key not in keys:
            continue
        r_obj = r['readRDS'](str(f))
        loaded[key] = r_obj

    if hasattr(pm, 'gam_models'):
        pm.gam_models.update(loaded)
    else:
        pm.gam_models = loaded
    return pm.gam_models
