#!/usr/bin/env python3
"""
Example: Using converted sensor log data for classification

This example demonstrates how to use converted recorded sensor logs
with the time-series classifier, just like synthetic data.
"""

import json
import os
from mcap.reader import make_reader


def load_sensor_log_data(base_name: str, data_dir: str = "./temp"):
    """
    Load converted sensor log data (MCAP, labels, and experiment config)
    
    Args:
        base_name: Base name of the sensor log (e.g., "Sensor_Log_sn3360_2025-10-03_01_21_10")
        data_dir: Directory containing the converted files
    
    Returns:
        dict: Dictionary with 'mcap_file', 'mcap_summary', 'labels', and 'experiment' keys
    """
    
    mcap_file = os.path.join(data_dir, f"{base_name}.mcap")
    labels_file = os.path.join(data_dir, f"{base_name}.labels.json")
    experiment_file = os.path.join(data_dir, f"{base_name}.experiment.json")
    
    # Read MCAP file summary
    print(f"Loading MCAP file: {mcap_file}")
    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
    
    # Load labels
    print(f"Loading labels: {labels_file}")
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    # Load experiment config
    print(f"Loading experiment config: {experiment_file}")
    with open(experiment_file, 'r') as f:
        experiment = json.load(f)
    
    return {
        'mcap_file': mcap_file,
        'mcap_summary': summary,
        'labels': labels,
        'experiment': experiment
    }


def print_summary(data: dict):
    """Print a summary of the loaded sensor log data"""
    
    print("\n" + "="*60)
    print("Sensor Log Data Summary")
    print("="*60)
    
    # Experiment metadata
    metadata = data['experiment']['metadata']
    print(f"\nDevice Information:")
    print(f"  Model: {metadata.get('deviceModel', 'N/A')}")
    print(f"  Serial: {metadata.get('serialNumber', 'N/A')}")
    print(f"  Firmware: {metadata.get('firmware', 'N/A')}")
    print(f"  OS: {metadata.get('osVersion', 'N/A')}")
    
    # Duration
    print(f"\nRecording Duration:")
    print(f"  Total: {metadata.get('total_duration_s', 0):.2f} seconds")
    
    # Available channels
    if data['mcap_summary'] and data['mcap_summary'].channels:
        channels = data['mcap_summary'].channels
        print(f"\nAvailable Sensor Channels: {len(channels)}")
        for channel_id, channel in channels.items():
            print(f"  - {channel.topic} (id: {channel_id})")
    
    # Message counts
    if data['mcap_summary'] and data['mcap_summary'].statistics:
        stats = data['mcap_summary'].statistics
        print(f"\nMessage Statistics:")
        print(f"  Total messages: {stats.message_count}")
        print(f"  Start time: {stats.message_start_time} ns")
        print(f"  End time: {stats.message_end_time} ns")
    
    # Labels
    events = data['labels'].get('events', [])
    print(f"\nLabels:")
    print(f"  Number of events: {len(events)}")
    for i, event in enumerate(events):
        mag_dist = event['metadata'].get('mag_distortion', 'N/A')
        start = event['startTime']['sec']
        end = event['endTime']['sec']
        print(f"  Event {i+1}: mag_distortion={mag_dist}, duration={(end-start):.1f}s")
    
    print("="*60)


def main():
    """Example usage of converted sensor log data"""
    
    # Load the first sensor log
    base_name = "Sensor_Log_sn3360_2025-10-03_01_21_10"
    
    print(f"Loading converted sensor log: {base_name}")
    print("-" * 60)
    
    try:
        data = load_sensor_log_data(base_name)
        print_summary(data)
        
        print("\n✓ Successfully loaded and processed sensor log data!")
        print("\nThis data can now be used for:")
        print("  - Training classifiers with fine_tune.py")
        print("  - Running classification with classify.py")
        print("  - Analysis in Jupyter notebooks")
        print("  - Visualization and exploration")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you've run convert_sensor_logs.py first:")
        print("  python scripts/convert_sensor_logs.py")
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
