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
    base_plan = load_base_plan()
    
    plans = []
    
    # Plan 1: No magnetic distortion (baseline)
    plan_1 = json.loads(json.dumps(base_plan))  # Deep copy
    plan_1["segments"] = [
        {
            "name": "clean_roll",
            "duration_s": 5.0,
            "rotation_rpy_degrees": {"roll": 45.0, "pitch": 0.0, "yaw": 0.0},
            "mag_distortion": {
                "level": "none",
                "world_x": 0.0,
                "world_y": 0.0,
                "world_z": 0.0
            }
        },
        {
            "name": "clean_pitch",
            "duration_s": 5.0,
            "rotation_rpy_degrees": {"roll": 0.0, "pitch": 45.0, "yaw": 0.0},
            "mag_distortion": {
                "level": "none",
                "world_x": 0.0,
                "world_y": 0.0,
                "world_z": 0.0
            }
        },
        {
            "name": "clean_yaw", 
            "duration_s": 5.0,
            "rotation_rpy_degrees": {"roll": 0.0, "pitch": 0.0, "yaw": 90.0},
            "mag_distortion": {
                "level": "none",
                "world_x": 0.0,
                "world_y": 0.0,
                "world_z": 0.0
            }
        }
    ]
    plans.append(("no_distortion", plan_1))
    
    # Plan 2: Low magnetic distortion
    plan_2 = json.loads(json.dumps(base_plan))  # Deep copy
    plan_2["segments"] = [
        {
            "name": "low_distortion_roll",
            "duration_s": 5.0,
            "rotation_rpy_degrees": {"roll": 45.0, "pitch": 0.0, "yaw": 0.0},
            "mag_distortion": {
                "level": "low",
                "world_x": 10.0,  # μT distortion in world X
                "world_y": 5.0,   # μT distortion in world Y
                "world_z": 0.0
            }
        },
        {
            "name": "low_distortion_pitch",
            "duration_s": 5.0,
            "rotation_rpy_degrees": {"roll": 0.0, "pitch": 45.0, "yaw": 0.0},
            "mag_distortion": {
                "level": "low",
                "world_x": 0.0,
                "world_y": 8.0,   # μT distortion in world Y
                "world_z": 6.0    # μT distortion in world Z
            }
        },
        {
            "name": "low_distortion_yaw",
            "duration_s": 5.0,
            "rotation_rpy_degrees": {"roll": 0.0, "pitch": 0.0, "yaw": 90.0},
            "mag_distortion": {
                "level": "low",
                "world_x": 5.0,
                "world_y": 0.0,
                "world_z": 10.0
            }
        }
    ]
    plans.append(("low_distortion", plan_2))
    
    # Plan 3: High magnetic distortion
    plan_3 = json.loads(json.dumps(base_plan))  # Deep copy
    plan_3["segments"] = [
        {
            "name": "high_distortion_roll",
            "duration_s": 5.0,
            "rotation_rpy_degrees": {"roll": 45.0, "pitch": 0.0, "yaw": 0.0},
            "mag_distortion": {
                "level": "high",
                "world_x": 25.0,  # μT distortion in world X
                "world_y": 15.0,  # μT distortion in world Y
                "world_z": 0.0
            }
        },
        {
            "name": "high_distortion_pitch", 
            "duration_s": 5.0,
            "rotation_rpy_degrees": {"roll": 0.0, "pitch": 45.0, "yaw": 0.0},
            "mag_distortion": {
                "level": "high",
                "world_x": 0.0,
                "world_y": 30.0,  # μT distortion in world Y
                "world_z": 20.0   # μT distortion in world Z
            }
        },
        {
            "name": "high_distortion_yaw",
            "duration_s": 5.0,
            "rotation_rpy_degrees": {"roll": 0.0, "pitch": 0.0, "yaw": 90.0},
            "mag_distortion": {
                "level": "high",
                "world_x": 20.0,
                "world_y": 0.0,
                "world_z": 25.0
            }
        }
    ]
    plans.append(("high_distortion", plan_3))
    
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
        mcap_file = f"data/{plan_name}.mcap"
        labels_file = f"data/{plan_name}.labels.json"
        
        generator.generate_from_plan_data(plan_data, mcap_file, labels_file)
        
        # Upload to Nstrumenta
        remote_prefix = f"synthetic_datasets/{plan_name}"
        upload_with_prefix(nst_client, mcap_file, remote_prefix)
        upload_with_prefix(nst_client, labels_file, remote_prefix)
        
        # Create experiment configuration for this dataset
        experiment_config = {
            "dirname": f"synthetic_datasets/{plan_name}",
            "dataFilePath": f"projects/nst-test/data/synthetic_datasets/{plan_name}/{mcap_file}",
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