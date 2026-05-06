# Menstrual Cycle Physiology, Sleep, and Wearable Data — Reproduction Code

Companion code for *"The menstrual cycle through the lens of a wearable device: insights into physiology, sleep, and cycle variability"* (npj Digital Medicine, 2026). Reproduces every main figure and statistical result in the paper.

## Layout

- `menstrual_cycle_analysis/` — minimal analysis package.
- `notebooks/` — five figure-family notebooks; each loads data once via the package and produces its figures.
  - `01_cycle_length.ipynb` (Fig 1, S1–S4) — cycle length × age, BMI, workout
  - `02_sleep_cycle_length.ipynb` (Fig 2, S5) — sleep × cycle length and deviation
  - `03_biometrics_gam.ipynb` (Fig 3, S6–S8, S10–S13) — biometric GAMs (R + rpy2 + mgcv)
  - `04_var_residuals.ipynb` (S9) — VAR(3) inter-biometric residuals
  - `05_sleep_phase_natural_experiment.ipynb` (Fig 4, S14) — within-subject sleep changes by phase
- `figures/` — generated paper figures (committed to the repo).
- `data/` — gitignored. Raw CSVs go here. See *Data* below.

## Install

### Recommended: conda

```bash
conda env create -f environment.yml
conda activate menstrual_cycle
```

This installs Python 3.11, the scientific stack, JupyterLab, and (optionally) R 4.x + rpy2 + mgcv for notebook 03. Comment out the `r-base`, `r-mgcv`, `rpy2` lines in `environment.yml` if you don't need notebook 03.

### Alternative: pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[notebook]              # core + jupyter
pip install -e .[notebook,r]            # add rpy2 (you must install R 4.x + mgcv yourself)
```

## Data

The two CSV files are private under the WHOOP–Stanford data agreement and are not committed:

- `data/physio_sleep_mc_power_users_daily_data.csv` — daily biometric / sleep / workout time-series, already cycle-aligned.
- `data/power_users_summary_table.csv` — per-user demographics and cycle summary statistics.

Place them under `data/` after access is granted. Contact the corresponding author for access.

## Running

After activating the environment and placing the data files:

```bash
jupyter lab notebooks/
```

The first call to `load_paper_data()` parses the 1.4M-row CSV (~30 s) and caches three parquet files under `data/cache/` for fast subsequent loads.

## Citation

```
Gonzalez A., O'Day J. J., Johnson S. C., Kim J., Jasinski S., Holmes K., Delp S. L., Hicks J. L.
"The menstrual cycle through the lens of a wearable device: insights into physiology, sleep,
and cycle variability." npj Digital Medicine, 2026.
```

## License

MIT — see `LICENSE`.
