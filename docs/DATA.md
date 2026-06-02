# Data schemas and conventions

The notebooks and `make_figures.py` expect two CSVs under `data/` (gitignored). Their schemas follow the time-series construction described in the paper Methods. The default filenames are configurable: `load_paper_data()` accepts `data_dir=`, `daily_csv=`, and `summary_csv=` to point at files in a different directory or under different names.

See the [README](../README.md) for how to obtain the data.

## Daily time-series CSV

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

## Per-participant summary CSV

One row per participant, indexed by `n_id`.

| Column | Description |
|---|---|
| `n_id` | Integer participant ID (matches the daily CSV) |
| `age` | Age in years at the start of data collection |
| `BMI` | Body mass index |

Participants whose median cycle length is outside 21–35 days, or who have fewer than 5 valid cycles, are dropped automatically (`valid_user` filter), matching the inclusion criteria in the paper.

## Filter and aggregation conventions

The package preserves the original analysis conventions:

- **Cycle validity (`vcl`)** — a cycle is valid if its length is in 15–45 days and both its bounds are inside the recording window.
- **Biometric filtering** — `process_physio_data(preset='biometric')` applies a per-participant zero-phase IIR bandpass (w₀=1/90, w₁=1/7) then percent-deviation normalization against each participant's mean. Three presets are available; see `PhysioMethods.FILTER_PRESETS`.
- **Workout load (eTRIMP)** — Σ *i* · *z<sub>i</sub>* over heart-rate zones 0–5 (Edwards' TRIMP, per the paper Methods).
- **Sleep variability** — log₂ variance of nightly `sleep_dur`, with the linear and quadratic terms used in cycle-length GEEs.
