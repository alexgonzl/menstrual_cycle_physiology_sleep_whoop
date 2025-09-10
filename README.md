# Menstrual Cycle Physiology Sleep Analysis

A Python package for analyzing menstrual cycle physiology and sleep data from wearable devices. This package provides the analysis code for the research paper: *"The menstrual cycle through the lens of a wearable device: insights into physiology, sleep, and cycle variability"*.

## Features

- **Cycle Length Analyses**: Generation of analysis tables by user and cycle with corresponding covariates
- **Data Processing**: Load, clean, and preprocess daily biometric data
- **Statistical Analysis**: Core approaches used in the analyses
- **Jupyter Notebooks**: Demonstration notebooks that generates paper figures, statistical results and a demo of the biometric GAM through the cycle.

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/alexgonzl/menstrual_cycle_physiology_sleep_whoop.git
cd menstrual_cycle_physiology_sleep_whoop

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

## Notebooks

The `notebooks/` directory contains demonstration notebooks:

- `paper_results.ipynb`: showcases code used for figure generation and statistical results as in the paper.
- `biometric_gam_cycle.ipynb`: loads statistical model and provides widgets to evaluate the model at tidfferent levels. 

To run the notebooks:
```bash
jupyter notebook notebooks/
```

## Contributing

This is a research package designed for reproducibility of published analyses. For questions or issues, please open a GitHub issue.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```
"The menstrual cycle through the lens of a wearable device: insights into physiology, sleep, and cycle variability"
[Full citation available when paper is published]
```
