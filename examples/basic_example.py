#!/usr/bin/env python3
"""
Example: Basic synthetic data generation and analysis
"""

import os
import sys

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from synthetic import SyntheticDataGenerator
from mcap_utils import read_synthetic_sensor_data, extract_imu_windows


def main():
    """Demonstrate basic synthetic data workflow"""
    
    print("=== Basic Synthetic Data Example ===\n")
    
    # Configuration
    plan_file = "config/default_plan.json"
    output_file = "examples/basic_synthetic.mcap"
    
    # Step 1: Generate synthetic data
    print("1. Generating synthetic data...")
    generator = SyntheticDataGenerator()
    generator.generate(plan_file, output_file)
    
    # Step 2: Read and analyze the data
    print("\n2. Reading synthetic sensor data...")
    data = read_synthetic_sensor_data(output_file)
    
    print("   Data summary:")
    for channel, samples in data.items():
        if samples:
            duration = (samples[-1][0] - samples[0][0]) / 1e9
            print(f"     {channel}: {len(samples)} samples over {duration:.1f}s")
    
    # Step 3: Extract windowed data
    print("\n3. Extracting windowed data...")
    windows = extract_imu_windows(output_file, window_size_ns=2e9, step_size_ns=1e9)
    print(f"   Extracted {len(windows)} windows (2s windows, 1s step)")
    
    if windows:
        print(f"   Example window at {windows[0]['timestamp']/1e9:.1f}s:")
        for channel, channel_data in windows[0]['window_data'].items():
            print(f"     {channel}: {len(channel_data)} samples")
    
    print(f"\n=== Example completed! ===")
    print(f"Generated file: {output_file}")


if __name__ == "__main__":
    main()
