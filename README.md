# Menstrual Cycle Physiology Sleep Analysis

A Python package for analyzing menstrual cycle physiology and sleep data from wearable devices. This package provides the analysis code for the research paper: *"The menstrual cycle through the lens of a wearable device: insights into physiology, sleep, and cycle variability"*.

## Features

- **Data Processing**: Load, clean, and preprocess physiological data from wearable devices
- **Sleep Analysis**: Analyze sleep patterns, duration, efficiency, and quality across menstrual cycle phases
- **Cycle Analysis**: Identify cycle phases, analyze variability, and detect ovulation patterns
- **Statistical Analysis**: Compare metrics across cycle phases with statistical testing
- **R Integration**: Optional integration with R for advanced statistical analyses
- **Jupyter Notebooks**: Demonstration notebooks showing package usage and research results

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
```

### With R Integration (Optional)

For advanced statistical features using R:

```bash
# Install R dependencies
pip install rpy2

# Ensure R is installed on your system with required packages
R -e "install.packages(c('stats', 'forecast', 'TSA'))"
```

## Quick Start

```python
from menstrual_cycle_analysis import DataProcessor, SleepAnalyzer, CycleAnalyzer
import pandas as pd
from datetime import datetime

# Load your data
processor = DataProcessor()
data = processor.load_data('your_sleep_data.csv')

# Analyze sleep patterns
sleep_analyzer = SleepAnalyzer()
sleep_analyzer.load_sleep_data(data)
sleep_metrics = sleep_analyzer.calculate_sleep_metrics()

# Analyze cycle patterns
cycle_analyzer = CycleAnalyzer()
cycle_analyzer.load_cycle_data(data)

# Define menstruation dates
menstruation_dates = [datetime(2024, 1, 1), datetime(2024, 1, 29)]
cycle_phases = cycle_analyzer.identify_cycle_phases(menstruation_dates)

# Analyze sleep by cycle phase
sleep_by_phase = sleep_analyzer.analyze_sleep_by_cycle_phase(cycle_phases)
```

## Notebooks

The `notebooks/` directory contains demonstration notebooks:

- `01_basic_usage_example.ipynb`: Basic package usage and analysis workflows
- `02_r_integration_example.ipynb`: Advanced statistical analysis using R integration

To run the notebooks:

```bash
jupyter notebook notebooks/
```

## Package Structure

```
menstrual_cycle_analysis/
├── __init__.py              # Main package imports
├── data_processing.py       # Data loading and preprocessing
├── sleep_analysis.py        # Sleep pattern analysis
└── cycle_analysis.py        # Menstrual cycle analysis
```

## Data Format

The package expects data in CSV format with the following columns:

**Required columns:**
- `date`: Date in YYYY-MM-DD format
- `sleep_duration`: Sleep duration in hours
- `sleep_efficiency`: Sleep efficiency percentage

**Optional columns:**
- `rem_sleep`: REM sleep duration in hours
- `deep_sleep`: Deep sleep duration in hours
- `heart_rate`: Average heart rate in BPM
- `body_temperature`: Body temperature
- Additional physiological metrics

## Contributing

This is a research package designed for reproducibility of published analyses. For questions or issues, please open a GitHub issue.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```
"The menstrual cycle through the lens of a wearable device: insights into physiology, sleep, and cycle variability"
[Add full citation when published]
```
