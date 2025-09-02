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
    """Create different plan variations for diverse training data"""
    base_plan = load_base_plan()
    
    plans = []
    
    # Plan 1: Quick rotation test - fast movements
    plan_1 = json.loads(json.dumps(base_plan))  # Deep copy
    plan_1["segments"] = [
        {
            "name": "rapid_roll",
            "duration_s": 3.0,
            "rotation_rpy_degrees": {"roll": 90.0, "pitch": 0.0, "yaw": 0.0}
        },
        {
            "name": "rapid_pitch",
            "duration_s": 3.0,
            "rotation_rpy_degrees": {"roll": 0.0, "pitch": 90.0, "yaw": 0.0}
        },
        {
            "name": "rapid_yaw", 
            "duration_s": 3.0,
            "rotation_rpy_degrees": {"roll": 0.0, "pitch": 0.0, "yaw": 180.0}
        }
    ]
    plans.append(("quick_rotation", plan_1))
    
    # Plan 2: Slow precise movements
    plan_2 = json.loads(json.dumps(base_plan))  # Deep copy
    plan_2["segments"] = [
        {
            "name": "slow_roll",
            "duration_s": 20.0,
            "rotation_rpy_degrees": {"roll": 45.0, "pitch": 0.0, "yaw": 0.0}
        },
        {
            "name": "slow_pitch",
            "duration_s": 20.0,
            "rotation_rpy_degrees": {"roll": 0.0, "pitch": 45.0, "yaw": 0.0}
        },
        {
            "name": "slow_combined",
            "duration_s": 20.0,
            "rotation_rpy_degrees": {"roll": 30.0, "pitch": 30.0, "yaw": 30.0}
        }
    ]
    plans.append(("slow_precise", plan_2))
    
    # Plan 3: Complex multi-axis movements
    plan_3 = json.loads(json.dumps(base_plan))  # Deep copy
    plan_3["segments"] = [
        {
            "name": "complex_1",
            "duration_s": 8.0,
            "rotation_rpy_degrees": {"roll": 45.0, "pitch": 30.0, "yaw": 60.0}
        },
        {
            "name": "complex_2", 
            "duration_s": 8.0,
            "rotation_rpy_degrees": {"roll": -30.0, "pitch": 45.0, "yaw": -45.0}
        },
        {
            "name": "complex_3",
            "duration_s": 8.0,
            "rotation_rpy_degrees": {"roll": 60.0, "pitch": -30.0, "yaw": 90.0}
        },
        {
            "name": "return_to_origin",
            "duration_s": 6.0,
            "rotation_rpy_degrees": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        }
    ]
    plans.append(("complex_multi_axis", plan_3))
    
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
        
        # Generate synthetic data
        mcap_file = f"data/{plan_name}.mcap"
        generator.generate_from_plan_data(plan_data, mcap_file)
        
        # Upload to Nstrumenta
        remote_prefix = f"synthetic_datasets/{plan_name}"
        upload_with_prefix(nst_client, mcap_file, remote_prefix)
        
        # Create experiment configuration for this dataset
        experiment_config = {
            "dirname": f"synthetic_datasets/{plan_name}",
            "dataFilePath": f"projects/nst-test/data/synthetic_datasets/{plan_name}/{mcap_file}",
            "description": f"Synthetic dataset: {plan_name}",
            "segments": plan_data["segments"],
            "metadata": {
                "generated_by": "synthetic_data.py",
                "sample_rate": plan_data["initialization"]["sample_rate"],
                "total_duration_s": sum(seg["duration_s"] for seg in plan_data["segments"])
            }
        }
        
        # Save and upload experiment config
        experiment_file = f"data/{plan_name}.experiment.json"
        with open(experiment_file, 'w') as f:
            json.dump(experiment_config, f, indent=2)
        
        upload_with_prefix(nst_client, experiment_file, remote_prefix)
        
        print(f"  Generated: {mcap_file}")
        print(f"  Uploaded to: {remote_prefix}")
    
    print(f"\nSuccessfully generated {len(plans)} synthetic datasets!")
    print("Datasets can be used for:")
    print("- Training classification models")
    print("- Testing motion detection algorithms") 
    print("- Validating sensor fusion techniques")


if __name__ == "__main__":
    main()