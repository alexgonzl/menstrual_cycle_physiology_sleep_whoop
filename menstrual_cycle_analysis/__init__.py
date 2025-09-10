"""
Menstrual Cycle Analysis Package

A Python package for analyzing menstrual cycle physiology and sleep data from wearable devices.
This package provides tools for processing and analyzing data as presented in:
"The menstrual cycle through the lens of a wearable device: insights into physiology, sleep, and cycle variability"
"""

__version__ = "0.1.0"
__author__ = "Alex Gonzalez"

# Import main analysis modules
from .data_processing import DataProcessor
from .sleep_analysis import SleepAnalyzer
from .cycle_analysis import CycleAnalyzer

__all__ = [
    "DataProcessor",
    "SleepAnalyzer", 
    "CycleAnalyzer"
]