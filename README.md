# Menstrual Cycle Physiology, Sleep, and Wearable Data — Reproduction Code

Companion code for *"The menstrual cycle through the lens of a wearable device: insights into physiology, sleep, and cycle variability"* (npj Digital Medicine, 2026). Reproduces every main and supplementary figure plus the manuscript-quoted statistics, given the two CSVs described under [Data](#data).

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
- `models/` — gitignored. Notebook 03 caches the fitted `mgcv::bam` GAMs as `.rds` here on first run; subsequent runs are ~10× faster.

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

The original WHOOP–Stanford CSVs are private under the data-use agreement and are not redistributed. Anyone with comparable wearable + cycle data can reproduce the analyses by providing two CSVs that match the schema below.

### Daily time-series CSV

One row per (subject, day). Required columns:

| Column | Description |
|---|---|
| `n_id` | Integer subject ID |
| `date` | Date string parseable by `pd.to_datetime` |
| `day` | Integer index of the day within the subject's record (monotonically increasing) |
| `starts` | 0/1 — set to 1 on the first day of each menstrual cycle (i.e., menstrual onset), 0 otherwise. The codebase uses these flags to derive `j_cycle`, `j_cycle_num`, and cycle-day indexing |
| `RHR` | Daily resting heart rate (beats/min) |
| `HRV` | Daily heart-rate variability (ms; root mean square of successive differences) |
| `RR` | Daily resting respiratory rate (breaths/min) |
| `skin_temp` | Daily skin-temperature deviation (°C) |
| `blood_oxygen` | Daily blood-oxygen saturation (%) |
| `sleep_dur` | Hours of sleep that night |
| `sl_onset` | Sleep-onset clock time (hours since midnight, 0–24, with wrap) |
| `time_in_bed` | Hours in bed |
| `wo_time_0`, `wo_time_1`, `wo_time_2` | Start times of the day's first, second, and third workout (hours since midnight). `NaN` if no workout |
| `intensity_0`, `intensity_1`, `intensity_2` | Average heart-rate intensity (au) of the corresponding workout. Workouts with intensity below 30 are dropped |
| `duration_0`, `duration_1`, `duration_2` | Workout duration (hours) |
| `z0`, `z1`, `z2`, `z3`, `z4`, `z5` | Minutes spent in each of six heart-rate zones across all activity that day. Used to compute eTRIMP = Σ i · z_i |

Missing biometric values are allowed (NaN), but `day` must be a continuous integer sequence per subject — the cycle-day pipeline indexes by row position, so gaps misalign cycles. Cycles that abut the recording boundary are dropped (`drop_unbounded_cycles`).

### Per-subject summary CSV

One row per subject, indexed by `n_id`. Required columns:

| Column | Description |
|---|---|
| `n_id` | Integer subject ID (matches the daily CSV) |
| `age` | Age in years (used as a continuous and a binned covariate) |
| `BMI` | Body mass index |

Subjects whose median cycle length falls outside 21–35 days, or who have fewer than 5 valid cycles, are dropped automatically (`valid_user` filter).

### Filter and aggregation conventions

The package preserves the original analysis conventions:

- **Cycle validity (`vcl`)** — a cycle is valid if its length is in 15–45 days and both its bounds are inside the recording window.
- **Biometric filtering** — `process_physio_data(preset='biometric')` applies a per-subject zero-phase IIR bandpass (w₀=1/90, w₁=1/7) then percent-deviation normalization. Three presets are available; see `PhysioMethods.FILTER_PRESETS`.
- **Workout load (eTRIMP)** — `Σ i · z_i` over heart-rate zones 0–5 (matches the manuscript Methods).
- **Sleep variability** — log₂ variance of nightly `sleep_dur`, with the linear and quadratic terms used in cycle-length GEEs.

## Running

After activating the environment and placing the data files:

```bash
jupyter lab notebooks/
```

The first call to `load_paper_data()` parses the daily CSV (~30 s for 1.4M rows) and caches three parquet files under `data/cache/` for fast subsequent loads. Notebook 03's first run also fits five GAMs (~30–60 minutes total); fits are cached as `models/pct_*.rds` so subsequent runs take ~5 minutes.

End-to-end execution of all five notebooks:

```bash
for nb in 01_cycle_length 02_sleep_cycle_length 03_biometrics_gam \
         04_var_residuals 05_sleep_phase_natural_experiment; do
    python -m nbconvert --to notebook --execute notebooks/$nb.ipynb \
        --output /tmp/$nb.ipynb --ExecutePreprocessor.timeout=7200
done
```

## Citation

```
Gonzalez A., O'Day J. J., Johnson S. C., Kim J., Jasinski S., Holmes K., Delp S. L., Hicks J. L.
"The menstrual cycle through the lens of a wearable device: insights into physiology, sleep,
and cycle variability." npj Digital Medicine, 2026.
```

## License

MIT — see `LICENSE`.
