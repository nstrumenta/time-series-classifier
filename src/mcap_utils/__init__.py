"""
MCAP utilities for time series data processing
"""

from .reader import (
    read_synthetic_sensor_data,
    extract_imu_windows,
    print_mcap_summary
)

from .visualization import (
    plot_synthetic_sensor_data
)

from .dataset import (
    create_dataset,
    serialize_numpy_array,
    deserialize_numpy_array
)

from .spectrogram import (
    spectrogram_from_timeseries,
    classify_from_spectrogram
)

__all__ = [
    'read_synthetic_sensor_data',
    'extract_imu_windows', 
    'print_mcap_summary',
    'plot_synthetic_sensor_data',
    'create_dataset',
    'serialize_numpy_array',
    'deserialize_numpy_array',
    'spectrogram_from_timeseries',
    'classify_from_spectrogram'
]
