# Handoff

Public reproduction codebase for *"The menstrual cycle through the lens of a wearable device"* (npj Digital Medicine, 2026). Built from the private sibling repo `whoop_analyses/` by porting the relevant slice and rewiring it to load the two public CSVs in `data/`.

## Status

**Done — committed and verified against the source notebook + manuscript:**

| Notebook | Figures | Phase | Commit |
|---|---|---|---|
| `01_cycle_length.ipynb` | Fig 1, S1, S2, S3, S4 | B | `43a2c93` |
| `02_sleep_cycle_length.ipynb` | Fig 2, S5 | C | `9928fb9` |
| `04_var_residuals.ipynb` | S9 | D-1 | `41a1f4c` |
| `05_sleep_phase_natural_experiment.ipynb` | Fig 4a, 4b, 4c, S14 | E-1 / E-2 / E-3 | `682da81`, `55b5acc`, `e698dee` |

**Deferred:**

- `03_biometrics_gam.ipynb` (Fig 3 + S6, S7, S8, S10, S11, S12, S13) — requires porting the GAM machinery (`PhysioMethods` GAM-fitting methods + `mgcv::bam` via rpy2 + S10 critical-value heatmaps + S11 magnitude-range models + S12 individual cycle plots + S13 sleep-binned biometrics). Largest single port; ~3000 LOC.
- Public `CLAUDE.md` for attribution — see "Loose ends" below.

## Numerical match against the source

For every committed figure, the numerical match was verified against the
source notebook saved outputs and/or the manuscript-quoted statistics:

- **Fig 1**: 2,596 users / 42,759 cycles / 1,298,555 data days. Conditional CL: 29.11 @ age 24, 26.88 @ age 44, contrast 2.23. Deviation min at age 33 = 0.245.
- **Fig 2**: within-subject OR for premenstrual sleep ≥10% reduction = 1.39 [1.15, 1.69] (manuscript: 1.39 [1.14, 1.68]).
- **Fig 4 / S14**: phase counts 2,312 subjects / 47,702 obs / 20,814 cycles match cell 100. Per-biometric model counts 2,312 / 47,692 / 20,806 (cardiorespiratory) and 2,288 / 40,441 / 17,927 (skin temp) match cell 102.
- **S9 (VAR)**: residual correlation matrix and FEVD match the manuscript text (RHR explains 49% of HRV, 18% of RR, 3% of skin temp).

## Repo layout

```
menstrual_cycle_analysis/
  config.py                       constants (bin edges, paths, biometrics)
  io.py                           load_paper_data() entry point
  cl_behav_methods.py             CycleBehavMethods (cycle/user tables, sleep/wo aggregates, plot_var_x_*)
  paper_code_wrapper.py           CycleLengthAnalyses (Fig 1/2/S1-S5 figures + stats + GEE wrapper)
  physio_methods.py               PhysioMethods (biometric filters, reference table, process_physio_data)
                                  + PhysioBehavChangeMethods (Fig 4a histograms, behavior-change data prep)
  physio_behavior_analyses.py     PhysioBehaviorAnalyses (Fig 4a/b/c, S14)
  pca_analyses.py                 Biometrics_VAR (S9)
  plotting_routines.py            PLOTTING_ROUTINES utility (kept untouched)
  _plot_utils.py                  setup_axes, single_var_point_plot, fixed_yticks (helpers)
  stats/
    circular.py                   circmean_day, circvar_day, mad
    contrasts.py                  StatisticalPredictionHandler (conditional / within-subject contrasts, min_term_ci)
notebooks/
  01_cycle_length.ipynb           Fig 1, S1, S2, S3, S4
  02_sleep_cycle_length.ipynb     Fig 2, S5
  04_var_residuals.ipynb          S9
  05_sleep_phase_natural_experiment.ipynb   Fig 4 a/b/c, S14
figures/                          generated SVG + PNG (committed)
data/                             gitignored — private CSVs
models/                           gitignored — reserved for cached R objects
```

Total: ~4,855 LOC across 13 Python files.

## Setup

```bash
# Recommended (handles R + rpy2 cleanly for notebook 03 if you go that route)
conda env create -f environment.yml
conda activate menstrual_cycle

# Pip-only path (notebooks 01/02/04/05 only)
python -m venv .venv && source .venv/bin/activate
pip install -e .[notebook]
```

CSVs go under `data/`:

- `physio_sleep_mc_power_users_daily_data.csv`
- `power_users_summary_table.csv`

## Running

```bash
jupyter lab notebooks/
```

First call to `load_paper_data()` parses the 1.4M-row CSV (~30s). Subsequent
loads use parquet caches under `data/cache/`.

## How the port was done

**The source notebook (`whoop_analyses/notebooks/paper_analyses/figure_generator_revision.ipynb`) is the source of truth.** The cell sequence drives state. The supporting `.py` files (`cl_behav_methods.py`, `physio_methods.py`, `paper_code_wrapper.py`) are messy and contain branches/methods that don't run for the published figures — they should never be used to infer behavior.

For each notebook, the port follows the source notebook's cell sequence verbatim, including subtle setup steps (e.g., notebook 05 needs a "throwaway" first PBA construction at source-cell 68 that mutates `PM` state before the actual PBA at cell 91).

Method bodies are byte-identical to the source where possible. The only mods:
- `WHOOP_USER_TABLES` private dependency replaced with `(day_df, summary_df)` arguments.
- `paper_figures_path` set from `config.FIGURES_DIR` instead of `wt.paper_figures_path`.
- `self.wt.HR_ZONES` etc. replaced with `config.HR_ZONES`.
- `cardiovascular` activity-category column dropped (not used for any port-scope figure).

## Picking up notebook 03

To do Fig 3 + S6-S13:

1. **Source cells**: 25, 27, 66-85, 126-175 of `figure_generator_revision.ipynb`. Mostly call methods on `PhysioMethods`.
2. **Methods to port** from `whoop_analyses/whoop_analyses/physio_methods.py`:
   - `fit_gam_models`, `prep_data_gam_cycle_model`, `gam_cycle_model_bam_full` (the rpy2 + mgcv core).
   - `get_gam_predictions`, `get_gam_cl_age_sim_data`, `get_gam_sim_data_for_vars`.
   - `plot_gam_biometrics_cl_age` (Fig 3).
   - `print_gam_age_contrast`, `print_gam_cl_contrast`.
   - `plot_user_level_biometrics_vs_x` (S6, S7, S13).
   - `plot_biometrics_cl_age` (S8).
   - `get_gam_critical_values_simple`, `get_gam_critical_values_cwt`, `plot_biometrics_critical_values_heatmaps2` (S10).
   - `model_biom_delta_x_cl_age`, `gam_biom_delta_x_cl_age`, `plot_max_min_biometric_cycle_age_cl_variation_gam` (S11).
   - `individual_biometrics_x_cl` (S12).
   - `print_sup_table2_within_cycle_metrics`, `print_biometric_cycle_variation` (text outputs).
3. **R dependency**: rpy2 + R 4.x + mgcv. Already optional in `pyproject.toml [r]` and conda `environment.yml`. The `r_utils.py` file in the source has saveRDS/readRDS wrappers worth porting verbatim for caching fitted GAMs to `models/`.
4. **Caching**: GAM fits are slow (5-30 min each). Cache to `models/*.rds` keyed by `(biometric, ar_lags, df_shape, formula_hash)`.

## Loose ends

- **Public `CLAUDE.md`**: project memory says to commit a short attribution at wrap-up. The current `CLAUDE.md` is gitignored because it points at private paths. To add: `git add -f CLAUDE.md` after writing a clean ~10-line attribution (or remove the gitignore entry for that one file).
- **Caching `PM.process_physio_data` output**: notebooks 04 + 05 spend ~5-10 min in `process_physio_data`. Adding parquet caching of `PM.data` after preprocessing would speed up iteration by ~10× on warm starts. Hash key would be `(daily_csv_mtime, preset, prefix, types)`.
- **`Biometrics_VAR` per-row drift**: cohort counts and FEVD numbers match the manuscript exactly, but the row-by-row `BV.data` values differ from the source notebook's saved cell-137 output. The figure and headline numbers are correct; the saved cell output may be stale. Documented as known and accepted.

## How to run all four notebooks end-to-end

```bash
for nb in 01_cycle_length 02_sleep_cycle_length 04_var_residuals 05_sleep_phase_natural_experiment; do
    python -m nbconvert --to notebook --execute notebooks/$nb.ipynb \
        --output /tmp/$nb.ipynb --ExecutePreprocessor.timeout=2400
done
```

Notebook 01 runs in ~1 min. 02 in ~2 min. 04 in ~5 min (process_physio_data). 05 in ~12 min (two process_physio_data + 5 GEEs).

## Plan file

`/Users/alex/.claude/plans/this-is-a-stale-glistening-fox.md` — written across phases. The current version covers Phase E (notebook 05) detail.
