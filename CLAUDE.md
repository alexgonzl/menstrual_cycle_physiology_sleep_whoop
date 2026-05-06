# CLAUDE.md — menstrual_cycle_physiology_sleep_whoop

Public reproduction codebase for *"The menstrual cycle through the lens of a wearable device: insights into physiology, sleep, and cycle variability"* (npj Digital Medicine, revision_April2026).

Research code. Scientific correctness > engineering polish.

---

## What This Project Does

Reproduces every main figure and statistical result from the paper using two pre-anonymized CSVs (gitignored under `data/`) and a small Python package. The full analysis lives in a private sibling repo (`whoop_analyses/`) tied to private raw data; this repo strips that down to the slice required for paper reproduction.

5 figure-family notebooks under `notebooks/`:

| Notebook | Figures | Stats backend |
|---|---|---|
| `01_cycle_length.ipynb` | Fig 1, S1, S2, S3, S4 | statsmodels GEE |
| `02_sleep_cycle_length.ipynb` | Fig 2, S5 | statsmodels GEE |
| `03_biometrics_gam.ipynb` | Fig 3, S6, S7, S8, S10–S13 | rpy2 + mgcv::bam |
| `04_var_residuals.ipynb` | S9 | statsmodels VAR |
| `05_sleep_phase_natural_experiment.ipynb` | Fig 4, S14 | statsmodels GEE on a within-subject subset |

R is required only for notebook 03. Notebooks 01, 02, 04, 05 are pure Python.

---

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                # core (Python only — runs notebooks 01/02/04/05)
pip install -e .[r]             # add rpy2 (also requires R 4.x on PATH and `mgcv` installed)
```

Quick sanity check:
```bash
python -c "from menstrual_cycle_analysis.io import load_paper_data; d, c, u = load_paper_data(); print(d.shape, c.shape, u.shape)"
```

---

## Layout

```
menstrual_cycle_analysis/
  config.py            # constants: bin edges, thresholds, paths
  io.py                # load_paper_data() — entry point, parquet-cached
  cycles.py            # build_user_table, build_cycle_table, add_delta_thresholds
  behaviors.py         # day-level wo helpers + per-cycle sleep/workout aggregation
  plotting_routines.py # PLOTTING_ROUTINES utility (kept as-is)
  panels.py            # figure-panel composers (filled in per phase)
  stats/
    circular.py        # circmean_day, circvar_day, mad
    gee.py             # statsmodels GEE wrapper
    gam.py             # rpy2 + mgcv::bam wrapper (Phase D)
    var_model.py       # statsmodels VAR(3) (Phase E)
    contrasts.py       # GAM contrasts/CIs (Phase D)
    natural_experiment.py  # within-subject sleep-change subset (Phase C)
notebooks/             # one per figure family (Phase B onward)
data/                  # gitignored. Place CSVs here. See README for access.
data/cache/            # gitignored parquet caches of day_df, cycle_df, user_df
models/                # gitignored .rds files for cached mgcv fits
figures/               # COMMITTED. Generated SVG/PNG go here.
```

---

## Stats backend mapping (matches the paper)

- **GEE** for cycle-length and sleep-cycle-length figures (1, 2, 4 + supps): `statsmodels.formula.api.gee` with `Exchangeable` covariance, formula strings, observation weights = `1/log(N_in_bin)`.
- **GAM** for biometric × age × cycle-length surfaces (Fig 3 + supps): `mgcv::bam` via rpy2, exact formula ported verbatim from `whoop_analyses/whoop_analyses/physio_methods.py:535-568`. AR(1) within mgcv via `bam(..., AR.start=ar_start_logical, rho=0.3)`. Cached via `saveRDS` keyed by hash of (biometric, ar_lags, covariates, df shape).
- **VAR(3)** for inter-biometric coupling (S9): `statsmodels.tsa.api.VAR(...).fit(maxlags=3, trend='ct')`. Per-user normalize → concat → fit. Ported verbatim from `whoop_analyses/whoop_analyses/pca_analyses.py:306-345`.

---

## Working Principles

### 1. Port, don't re-derive
This repo is a simplification of the validated `whoop_analyses/` codebase. Default to copying functions verbatim and trimming dependencies. Rewrite only when (a) the source touches private internals, (b) the source is genuinely buggy/dead, or (c) the source's class structure doesn't fit the new pure-function layer. **Don't re-derive logic from manuscript text** — the source code is ground truth.

### 2. Numbers reproduce the paper
Any drift from published numbers is a bug. The validity rule is "every day in the cycle has `flags == 0`" (matches `src.py:1047`); the eTRIMP definition is `sum(i * z_i)` over HR zones (matches `cl_behav_methods.py:239`); the GEE family/cov-struct is `Gaussian + Exchangeable` (matches `paper_code_wrapper.py:105-110`).

### 3. Simplicity first
Minimum code that solves the problem. Pure functions over stateful classes. Notebooks are thin — one import, one `load_paper_data()` call, then panel composers do the work.

### 4. Surgical changes
Touch only what the current task requires. Don't refactor `plotting_routines.py` — it works. Don't touch `notebooks/` files outside the one currently being implemented.

### 5. Stage gates exist for a reason
Phase A landed the package foundation. Phase B is notebook 01 alone — stop and verify Fig 1 matches the manuscript before continuing to 02.

---

## Standing Rules

- **`figures/` is committed** to the repo. `data/`, `data/cache/`, and `models/` are gitignored.
- **Heatmaps are PNG-only** at 500 DPI. Other figures: SVG + PNG at 500 DPI.
- **Don't commit data**. The two CSVs are private under the WHOOP data agreement. README documents the access path.
- **R is optional**. The package must be importable on a pip-only machine; rpy2 is lazy-imported inside `stats/gam.py` and `stats/contrasts.py`.
- **No new top-level files without a reason.** This repo's whole job is reproduction; new modules need a clear figure or stat they're feeding.
- **Set numpy random seed** at the top of every notebook (`np.random.seed(0)`) for reproducible bootstrap CIs in `PLOTTING_ROUTINES.single_var_point_plot`.

---

## Reference paths (private, read-only)

- Source codebase: `/Users/alex/Documents/Research/WHOOP/whoop_analyses/`
- Master figure-generator notebook: `whoop_analyses/notebooks/paper_analyses/figure_generator_revision.ipynb`
- Manuscript: `/Users/alex/Documents/Research/WHOOP/Physio_paper/npj_DM/revision_April2026/GonzalezODay_revision2_final.docx` (and SM)

When in doubt about a function's behavior or formula, read the source there first; the manuscript is for sanity-checking final numbers, not for re-deriving methods.
