#!/usr/bin/env python3
"""
Generate synthetic sensor datasets for training and testing
"""

import json
import os
from script_utils import (
    init_script_environment,
    setup_working_directory,
    reset_to_initial_cwd,
    fetch_nstrumenta_file,
    upload_with_prefix,
)

# Initialize script environment
src_dir, nst_client = init_script_environment()

# Import project modules after src path is set up
import mcap_utilities
from synthetic import SyntheticDataGenerator


def load_base_plan():
    """Load the default plan as a base template"""
    base_plan_path = os.path.join(src_dir, "..", "config", "default_plan.json")
    with open(base_plan_path, 'r') as f:
        return json.load(f)


def create_plan_variations():
    """Create different plan variations for magnetic distortion classification"""
    import random
    
    def create_long_sequence_plan(file_index, total_duration=600):
        """Create a 10-minute plan with varying periods of distortion"""
        base_plan = load_base_plan()
        
        segments = []
        current_time = 0
        
        while current_time < total_duration:
            # Random segment duration between 5-30 seconds
            segment_duration = random.uniform(5, 30)
            
            # Don't exceed total duration
            if current_time + segment_duration > total_duration:
                segment_duration = total_duration - current_time
            
            # Random distortion level (0=none, 1=low, 2=high)
            distortion_level = random.choice([0, 1, 2])
            
            # Create distortion parameters based on level
            if distortion_level == 0:  # No distortion
                mag_distortion_value = 0.0
                level_name = "none"
            elif distortion_level == 1:  # Low distortion
                mag_distortion_value = random.uniform(0.3, 0.7)
                level_name = "low"
            else:  # High distortion
                mag_distortion_value = random.uniform(1.5, 2.5)
                level_name = "high"
            
            # Random rotation in roll/pitch/yaw degrees
            segment = {
                "name": f"{level_name}_motion_{len(segments)}",
                "duration_s": segment_duration,
                "rotation_rpy_degrees": {
                    "roll": random.uniform(-45, 45),
                    "pitch": random.uniform(-30, 30), 
                    "yaw": random.uniform(-90, 90)
                },
                "magnetic_distortion": mag_distortion_value,
                "mag_distortion": {
                    "level": level_name
                }
            }
            
            segments.append(segment)
            current_time += segment_duration
        
        plan = {
            "initialization": base_plan["initialization"],
            "segments": segments
        }
        
        return plan
    
    plans = []
    
    # Generate 5 long sequence files
    for i in range(5):
        plan = create_long_sequence_plan(i)
        plans.append((f"training_sequence_{i}", plan))
    
    return plans


def main():
    """Generate synthetic datasets and prepare for training"""
    
    # Set up working directory
    working_folder = "./temp/synthetic_datasets"
    reset_to_initial_cwd()
    setup_working_directory(working_folder)
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Initialize generator
    generator = SyntheticDataGenerator()
    
    # Create plan variations
    plans = create_plan_variations()
    
    print(f"Generating {len(plans)} synthetic datasets...")
    
    for plan_name, plan_data in plans:
        print(f"\nGenerating dataset: {plan_name}")
        
        # Generate synthetic data and labels
        mcap_file = f"{plan_name}.mcap"
        labels_file = f"{plan_name}.labels.json"
        
        generator.generate_from_plan_data(plan_data, mcap_file, labels_file)
        
        # Upload to Nstrumenta
        remote_prefix = f"synthetic_datasets/{plan_name}"
        upload_with_prefix(nst_client, mcap_file, remote_prefix)
        upload_with_prefix(nst_client, labels_file, remote_prefix)
        
        # Create experiment configuration for this dataset
        experiment_config = {
            "dirname": f"synthetic_datasets/{plan_name}",
            "labelFiles": [
                {
                    "filePath": f"projects/nst-test/data/synthetic_datasets/{plan_name}/{labels_file}"
                }
            ],
            "description": f"Synthetic dataset: {plan_name}",
            "segments": plan_data["segments"],
            "metadata": {
                "generated_by": "synthetic_data.py",
                "sample_rate": plan_data["initialization"]["sample_rate"],
                "total_duration_s": sum(seg["duration_s"] for seg in plan_data["segments"]),
                "classification_type": "mag_distortion",
                "distortion_levels": list(set(seg.get("mag_distortion", {}).get("level", "none") for seg in plan_data["segments"]))
            }
        }
        
        # Save and upload experiment config
        experiment_file = f"data/{plan_name}.experiment.json"
        with open(experiment_file, 'w') as f:
            json.dump(experiment_config, f, indent=2)
        
        upload_with_prefix(nst_client, experiment_file, remote_prefix)
        
        print(f"  Generated: {mcap_file}")
        print(f"  Generated: {labels_file}")
        print(f"  Uploaded to: {remote_prefix}")
    
    print(f"\nSuccessfully generated {len(plans)} synthetic datasets for magnetic distortion classification!")
    print("Each dataset includes:")
    print("- MCAP sensor data files with simulated magnetic distortions")
    print("- JSON label files with mag_distortion classifications (none=0, low=1, high=2)")
    print("- Experiment configuration files linking data and labels")
    print("\nDatasets can be used for:")
    print("- Training magnetic distortion classifiers")
    print("- Testing distortion detection algorithms") 
    print("- Validating magnetometer calibration techniques")
    print("- Supervised learning for magnetic anomaly detection")


if __name__ == "__main__":
    main()