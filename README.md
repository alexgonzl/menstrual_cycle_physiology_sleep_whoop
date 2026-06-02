# Menstrual Cycle Physiology, Sleep, and Wearable Data — Reproduction Code

Companion code for *"The menstrual cycle through the lens of a wearable device: insights into physiology, sleep, and cycle variability"* (Gonzalez & O'Day et al., *npj Digital Medicine*, 2026; [nature.com/articles/s41746-026-02799-9](https://www.nature.com/articles/s41746-026-02799-9); preprint at [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.09.11.675620v1)). Reproduces every main and supplementary figure plus the manuscript-quoted statistics, given the two CSVs described under [Data](#data).

**Project website: [alexgonzl.github.io/menstrual_cycle_physiology_sleep_whoop](https://alexgonzl.github.io/menstrual_cycle_physiology_sleep_whoop/)**

## Layout

- `menstrual_cycle_analysis/` — analysis package.
- `notebooks/` — five figure-family notebooks; each loads data once via the package and produces its figures.
  - `01_cycle_length.ipynb` (Fig 1, S1–S4) — cycle length × age, BMI, workout
  - `02_sleep_cycle_length.ipynb` (Fig 2, S5) — sleep × cycle length and deviation
  - `03_biometrics_gam.ipynb` (Fig 3, S6–S8, S10–S13) — biometric GAMs (R + rpy2 + mgcv)
  - `04_var_residuals.ipynb` (S9) — VAR(3) inter-biometric residuals
  - `05_sleep_phase_natural_experiment.ipynb` (Fig 4, S14) — within-subject sleep changes by phase
- `figures/` — generated paper figures (committed to the repo).
- `data/` — gitignored. Place the two CSVs here. See [Data](#data).
- `models/` — gitignored. Notebook 03 caches the fitted `mgcv::bam` GAMs as `.rds` here.

## Install

### Recommended: conda

```bash
conda env create -f environment.yml
conda activate menstrual_cycle
```

This installs Python 3.11, the scientific stack, JupyterLab, and (optionally) R 4.x + rpy2 + mgcv for notebook 03. Remove the R-related lines from `environment.yml` if you don't need notebook 03.

### Alternative: pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[notebook]              # core + jupyter
pip install -e .[notebook,r]            # add rpy2 (you must install R 4.x + mgcv yourself)
```

After installing rpy2 in either path, install `mgcv` once inside R:

```r
install.packages('mgcv')
```

## Data

The data that support the findings of this study are available from WHOOP, Inc., but restrictions apply to the availability of these data and so they are not publicly available. Data are however available from the authors upon reasonable request and with permission of WHOOP, Inc.

The notebooks expect two CSVs under `data/`: a daily time-series CSV (one row per participant-day) and a per-participant summary CSV. Column schemas, validity filters, and aggregation conventions are documented in [`docs/DATA.md`](docs/DATA.md). The default filenames are configurable via `load_paper_data(data_dir=, daily_csv=, summary_csv=)`.

## Running

### Regenerate the published figures

```bash
python make_figures.py                 # all 18 figures into figures/
python make_figures.py fig3 figS9      # subset
python make_figures.py --list          # show available figure names
```

`make_figures.py` is the single entry point that writes to `figures/`, calling the same package methods as the notebooks.

### Notebooks

```bash
jupyter lab notebooks/
```

Each notebook reproduces one figure family.

## Citation

```
Gonzalez A., O'Day J. J., Johnson S. C., Kim J., Jasinski S. R., Holmes K. E., Delp S. L., Hicks J. L.
"The menstrual cycle through the lens of a wearable device: insights into physiology, sleep,
and cycle variability." npj Digital Medicine 9, 2799 (2026).
https://doi.org/10.1038/s41746-026-02799-9
```

## License

MIT — see `LICENSE`.
