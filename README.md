# Menstrual Cycle Physiology, Sleep, and Wearable Data — Reproduction Code

Companion code for *"The menstrual cycle through the lens of a wearable device: insights into physiology, sleep, and cycle variability"* (Gonzalez & O'Day et al., 2025; preprint at [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.09.11.675620v1); under review). Reproduces every main and supplementary figure plus the manuscript-quoted statistics, given the two CSVs described under [Data](#data).

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

The data that support the findings of this study are available from WHOOP, Inc., but restrictions apply to the availability of these data and so they are not publicly available. Data are however available from the authors upon reasonable request and with permission of WHOOP, Inc.

The notebooks expect two CSVs under `data/`. Their schemas follow the time-series construction described in the paper Methods. The default filenames are configurable: `load_paper_data()` accepts `data_dir=`, `daily_csv=`, and `summary_csv=` to point at files in a different directory or under different names.

### Daily time-series CSV

One row per (participant, day), constructed by integrating menstrual, behavioral, and biometric data for each participant.

| Column | Description |
|---|---|
| `n_id` | Integer participant ID |
| `date` | Date string parseable by `pd.to_datetime` |
| `day` | Integer index of the day within the participant's record (monotonically increasing) |
| `starts` | 0/1 — 1 on the first day of each menstrual cycle (cycle onset, derived from logged menstrual status), 0 otherwise. Used to derive `j_cycle`, `j_cycle_num`, and cycle-day indexing |
| `RHR` | Resting heart rate (beats/min) — weighted average of the sleep period, with higher weights toward the end of sleep and the estimated slow-wave windows |
| `HRV` | Heart-rate variability (ms) — RMSSD, weighted from the sleep period as for RHR |
| `RR` | Respiratory rate (breaths/min) — median of the breath-rate estimates from interbeat intervals across sleep |
| `skin_temp` | Skin temperature (°C) — median of skin-temperature readings during sleep |
| `blood_oxygen` | Blood oxygen saturation (%) — median of pulse-oximetry readings during sleep |
| `sleep_dur` | Sleep duration that night (hours) |
| `sl_onset` | Sleep onset clock time (hours since midnight, 0–24) |
| `time_in_bed` | Time in bed (hours) |
| `wo_time_0`, `wo_time_1`, `wo_time_2` | Start time (hours since midnight) of up to the day's first three workouts; `NaN` if absent |
| `intensity_0`, `intensity_1`, `intensity_2` | Average heart-rate intensity of the corresponding workout (au). Workouts below 30 are dropped |
| `duration_0`, `duration_1`, `duration_2` | Workout duration (hours) |
| `z0`–`z5` | Minutes that day in each of six heart-rate zones (z1 = 50–59% of max HR through z5 = 90–100%, per the paper Methods; z0 = below z1, contributes 0 to eTRIMP). Daily eTRIMP is Σ *i* · *z<sub>i</sub>* (Edwards' TRIMP) |

Biometrics tolerate `NaN`; gaps of ≤7 days in a participant's daily biometric series are linearly interpolated by `process_physio_data`. The cycle-day pipeline indexes by row position, so `day` must form a continuous integer sequence per participant — gaps will misalign cycles. Cycles abutting the recording boundary are dropped (`drop_unbounded_cycles`).

Note that the WHOOP 3.0 device collects RHR, HRV, and RR; the WHOOP 4.0 additionally collects skin temperature and blood oxygen, so participants who switched devices have `NaN` for those two columns over the WHOOP 3.0 portion of their record.

### Per-participant summary CSV

One row per participant, indexed by `n_id`.

| Column | Description |
|---|---|
| `n_id` | Integer participant ID (matches the daily CSV) |
| `age` | Age in years at the start of data collection |
| `BMI` | Body mass index |

Participants whose median cycle length is outside 21–35 days, or who have fewer than 5 valid cycles, are dropped automatically (`valid_user` filter), matching the inclusion criteria in the paper.

### Filter and aggregation conventions

The package preserves the original analysis conventions:

- **Cycle validity (`vcl`)** — a cycle is valid if its length is in 15–45 days and both its bounds are inside the recording window.
- **Biometric filtering** — `process_physio_data(preset='biometric')` applies a per-participant zero-phase IIR bandpass (w₀=1/90, w₁=1/7) then percent-deviation normalization against each participant's mean. Three presets are available; see `PhysioMethods.FILTER_PRESETS`.
- **Workout load (eTRIMP)** — Σ *i* · *z<sub>i</sub>* over heart-rate zones 0–5 (Edwards' TRIMP, per the paper Methods).
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
and cycle variability." bioRxiv 2025.09.11.675620 (2025).
https://doi.org/10.1101/2025.09.11.675620
```

## License

MIT — see `LICENSE`.
