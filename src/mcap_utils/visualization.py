"""
MCAP data visualization utilities
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple
from .reader import read_synthetic_sensor_data


def plot_synthetic_sensor_data(mcap_file: str, channels: Optional[List[str]] = None, 
                              time_range: Optional[Tuple[int, int]] = None):
    """
    Plot synthetic sensor data from MCAP file
    
    Args:
        mcap_file: Path to MCAP file
        channels: List of channel names to plot (default: truth channels)
        time_range: Tuple of (start_time, end_time) in nanoseconds (default: all data)
    """
    if channels is None:
        channels = ["mag_truth", "acc_truth", "gyro_truth"]
    
    data = read_synthetic_sensor_data(mcap_file, channels)
    
    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 3 * len(channels)), sharex=True)
    if len(channels) == 1:
        axes = [axes]
    
    for i, channel in enumerate(channels):
        if channel not in data or not data[channel]:
            continue
            
        timestamps = [item[0] for item in data[channel]]
        values = [item[1] for item in data[channel]]
        
        # Filter by time range if specified
        if time_range:
            start_time, end_time = time_range
            filtered_data = [(t, v) for t, v in zip(timestamps, values) if start_time <= t <= end_time]
            if filtered_data:
                timestamps, values = zip(*filtered_data)
            else:
                timestamps, values = [], []
        
        # Convert timestamps to seconds for plotting
        timestamps_sec = [t / 1e9 for t in timestamps]
        
        # Plot each axis
        if values:
            values_array = np.array(values)
            if values_array.shape[1] >= 3:  # Ensure we have at least 3 dimensions
                axes[i].plot(timestamps_sec, values_array[:, 0], label='X', alpha=0.7)
                axes[i].plot(timestamps_sec, values_array[:, 1], label='Y', alpha=0.7)
                axes[i].plot(timestamps_sec, values_array[:, 2], label='Z', alpha=0.7)
            
        axes[i].set_title(f'{channel.replace("_", " ").title()}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel('Value')
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    plt.show()
