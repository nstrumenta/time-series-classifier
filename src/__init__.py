"""
Time Series Classifier - Synthetic Data Generation and MCAP Utilities

This package provides tools for generating synthetic sensor data and processing MCAP files
for time series classification tasks.
"""

from . import synthetic
from . import mcap_utils

__version__ = "1.0.0"
__all__ = ["synthetic", "mcap_utils"]
