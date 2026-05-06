# Copilot Instructions for Menstrual Cycle Physiology Sleep Analysis

## Project Overview
This repository analyzes menstrual cycle physiology and sleep data from wearable devices. It is structured as a Python package with core analysis modules and demonstration Jupyter notebooks. The code supports reproducibility for published research.

## Key Components
- `menstrual_cycle_analysis/`: Main package. Contains modules for cycle length analysis, data processing, plotting, sleep analysis, and utilities.
- `notebooks/`: Example and demo notebooks. Use these for figure generation, statistical results, and model evaluation.
- `data/`: Contains sample and real datasets (e.g., `physio_sleep_mc_power_users_daily_data.csv`).

## Developer Workflows
- **Environment Setup**: Use Conda for reproducible environments. Install dependencies with `pip install -r requirements.txt` and the package in editable mode with `pip install -e .`.
- **Notebook Usage**: Run Jupyter notebooks from the repo root with `jupyter notebook notebooks/`. Ensure the repo is installed in the active environment for imports to work.
- **Testing/Debugging**: Use the demo notebooks in `notebooks/` for interactive testing and validation of analysis routines.
- **Data Loading**: Data is loaded via pandas from CSV files in `data/`. Analysis modules expect DataFrames with specific columns (see notebook examples).

## Patterns & Conventions
- **Modular Design**: Each analysis type (cycle, sleep, plotting) is a separate module. Import only what you need.
- **Class-Based APIs**: Core analysis routines are implemented as classes. Instantiate and use as shown in notebooks.
- **R Integration**: Some modules support optional R integration via `rpy2`. Control with the `use_r` argument in class constructors.
- **Figure Generation**: Use plotting routines in `plotting_routines.py` for publication-quality figures. See notebook usage for examples.

## Integration Points
- **External Dependencies**: Main dependencies are listed in `requirements.txt`. R integration requires `rpy2` and a working R installation.
- **Editable Install**: Always install the repo in editable mode (`pip install -e .`) to ensure local changes are reflected in notebooks.

## Example Usage
```python
from menstrual_cycle_analysis import cycle_length_analyses as cla
```

## Troubleshooting
- If you see `ModuleNotFoundError`, ensure the repo is installed in the current Python environment and the kernel is restarted.
- For TOML errors, check `pyproject.toml` for syntax issues before running `pip install -e .`.

## References
- See `README.md` for installation and usage details.
- See `notebooks/` for practical examples and workflow demonstrations.

---

*Update this file as project conventions evolve. Focus on actionable, project-specific guidance for AI coding agents.*
