"""
MCAP reading utilities
"""

import json
from mcap.reader import make_reader
from typing import List, Dict, Tuple, Optional
import numpy as np


def print_mcap_summary(mcap_summary):
    """Print a summary of MCAP file contents"""
    if mcap_summary is None:
        print("No summary available.")
        return

    print("MCAP File Summary:")
    for channel in mcap_summary.channels.items():
        print(f"  - Channel: {channel}")


def read_mcap(file_name: str):
    """Basic MCAP file reader - prints all messages"""
    with open(file_name, "rb") as f:
        reader = make_reader(f)
        for message in reader.iter_messages():
            json_str = message.data.decode("utf8").replace("'", '"')
            print(json_str)


def read_synthetic_sensor_data(mcap_file: str, channels: Optional[List[str]] = None) -> Dict[str, List[Tuple[int, List[float]]]]:
    """
    Read synthetic sensor data from MCAP file
    
    Args:
        mcap_file: Path to MCAP file
        channels: List of channel names to read (default: all sensor channels)
    
    Returns:
        Dictionary with channel names as keys and lists of (timestamp, values) tuples as values
    """
    if channels is None:
        channels = ["mag_truth", "acc_truth", "gyro_truth", "mag_raw", "acc_raw", "gyro_raw", "pose_truth"]
    
    data = {channel: [] for channel in channels}
    
    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        
        for schema, channel, message in reader.iter_messages():
            if channel.topic in channels:
                json_str = message.data.decode("utf8")
                json_data = json.loads(json_str)
                data[channel.topic].append((json_data["timestamp"], json_data["values"]))
    
    return data


def extract_imu_windows(mcap_file: str, window_size_ns: float = 1e9, 
                       step_size_ns: Optional[float] = None, 
                       channels: Optional[List[str]] = None) -> List[Dict]:
    """
    Extract windowed IMU data for machine learning
    
    Args:
        mcap_file: Path to MCAP file
        window_size_ns: Window size in nanoseconds (default: 1 second)
        step_size_ns: Step size in nanoseconds (default: same as window_size_ns)
        channels: List of channel names to extract (default: raw sensor channels)
    
    Returns:
        List of dictionaries with 'timestamp', 'window_data', and 'duration_ns'
    """
    if step_size_ns is None:
        step_size_ns = window_size_ns
    
    if channels is None:
        channels = ["mag_raw", "acc_raw", "gyro_raw"]
    
    data = read_synthetic_sensor_data(mcap_file, channels)
    
    # Find time range
    all_timestamps = []
    for channel_data in data.values():
        all_timestamps.extend([item[0] for item in channel_data])
    
    if not all_timestamps:
        return []
    
    start_time = min(all_timestamps)
    end_time = max(all_timestamps)
    
    windows = []
    current_time = start_time
    
    while current_time + window_size_ns <= end_time:
        window_start = current_time
        window_end = current_time + window_size_ns
        
        window_data = {}
        for channel in channels:
            # Extract data in this window
            channel_window = []
            for timestamp, values in data[channel]:
                if window_start <= timestamp < window_end:
                    channel_window.append((timestamp, values))
            window_data[channel] = channel_window
        
        windows.append({
            'timestamp': window_start,
            'window_data': window_data,
            'duration_ns': window_size_ns
        })
        
        current_time += step_size_ns
    
    return windows
