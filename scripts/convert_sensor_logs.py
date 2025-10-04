#!/usr/bin/env python3
"""
Convert recorded sensor log JSON files to MCAP format with labels and experiment configs
"""

import json
import os
import uuid
from mcap.writer import Writer
from script_utils import (
    init_script_environment,
    setup_working_directory,
    reset_to_initial_cwd,
    upload_with_prefix,
)

# Initialize script environment
src_dir, nst_client = init_script_environment()


def parse_sensor_log_json(json_file: str) -> dict:
    """Parse a sensor log JSON file and extract metadata and data"""
    with open(json_file, 'r') as f:
        log_data = json.load(f)
    
    # Extract metadata
    metadata = {
        "firmware": log_data.get("firmware", ""),
        "serialNumber": log_data.get("serialNumber", ""),
        "appVersion": log_data.get("appVersion", ""),
        "deviceModel": log_data.get("deviceModel", ""),
        "deviceName": log_data.get("deviceName", ""),
        "osVersion": log_data.get("osVersion", ""),
        "userLabel": log_data.get("userLabel", "")
    }
    
    # Extract data records
    data = log_data.get("data", [])
    
    # Find START and STOP events to determine time range
    start_ts = None
    stop_ts = None
    
    for record in data:
        if record.get("type") == "START":
            start_ts = record.get("ts")
        elif record.get("type") == "STOP":
            stop_ts = record.get("ts")
    
    # If no explicit START/STOP, use first and last timestamps
    if start_ts is None and len(data) > 0:
        start_ts = data[0].get("ts")
    if stop_ts is None and len(data) > 0:
        stop_ts = data[-1].get("ts")
    
    return {
        "metadata": metadata,
        "data": data,
        "start_ts": start_ts,
        "stop_ts": stop_ts
    }


def convert_to_mcap(parsed_data: dict, output_mcap: str):
    """Convert parsed sensor log data to MCAP format"""
    
    # Set up sensor type to channel mapping
    sensor_type_map = {
        "MAG_AUTOCAL": "mag_raw",
        "ACCEL_AUTOCAL": "acc_raw",
        "GYRO_AUTOCAL": "gyro_raw",
        "Q_MAG_ACCEL": "q_mag_accel",
        "Q_9AXIS": "q_9axis",
        "LINEAR_ACCEL": "linear_accel",
        "TEMPERATURE": "temperature"
    }
    
    # Collect unique sensor types present in data
    sensor_types = set()
    for record in parsed_data["data"]:
        sensor_type = record.get("type")
        if sensor_type in sensor_type_map:
            sensor_types.add(sensor_type)
    
    with open(output_mcap, "wb") as f:
        writer = Writer(f)
        
        # Register schema
        schema_data = {
            "title": "sensor_event",
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    }
                }
            }
        }
        
        schema_id = writer.register_schema(
            name="sensor_event",
            encoding="jsonschema",
            data=json.dumps(schema_data).encode()
        )
        
        # Register channels for each sensor type
        channel_ids = {}
        for sensor_type in sensor_types:
            channel_name = sensor_type_map[sensor_type]
            channel_id = writer.register_channel(
                topic=channel_name,
                message_encoding="json",
                schema_id=schema_id
            )
            channel_ids[sensor_type] = channel_id
        
        writer.start()
        
        # Write data records
        for record in parsed_data["data"]:
            sensor_type = record.get("type")
            
            # Skip non-sensor records (START, STOP, TIMESTAMP_FULL)
            if sensor_type not in sensor_type_map:
                continue
            
            timestamp = record.get("ts")
            values = record.get("values", [])
            
            # Create message
            message_data = {
                "id": sensor_type_map[sensor_type],
                "timestamp": timestamp,
                "values": values
            }
            
            serialized = json.dumps(message_data)
            
            # Convert timestamp from microseconds to nanoseconds for MCAP
            timestamp_ns = timestamp * 1000
            
            writer.add_message(
                channel_ids[sensor_type],
                log_time=timestamp_ns,
                publish_time=timestamp_ns,
                data=serialized.encode()
            )
        
        writer.finish()
    
    print(f"  Converted to MCAP: {output_mcap}")


def generate_labels(parsed_data: dict, labels_file: str, log_filename: str, 
                   collection_path: str = None, distortion_level: str = "0"):
    """Generate labels file with a single event for the entire recording
    
    Args:
        parsed_data: Parsed sensor log data
        labels_file: Output labels JSON file path
        log_filename: Original log filename for reference
        collection_path: Path for the collection field in labels
        distortion_level: Default distortion level (0=none, 1=low, 2=high)
    """
    
    if collection_path is None:
        collection_path = f"projects/nst-test/data/recorded_logs/{os.path.basename(labels_file)}"
    
    start_ts = parsed_data["start_ts"]
    stop_ts = parsed_data["stop_ts"]
    
    # Convert timestamps from microseconds to nanoseconds
    start_ts_ns = start_ts * 1000
    stop_ts_ns = stop_ts * 1000
    
    # Create single event for entire recording
    event = {
        "id": str(uuid.uuid4()),
        "startTime": {
            "sec": int(start_ts_ns // 1e9),
            "nsec": int(start_ts_ns % 1e9)
        },
        "endTime": {
            "sec": int(stop_ts_ns // 1e9),
            "nsec": int(stop_ts_ns % 1e9)
        },
        "metadata": {
            "mag_distortion": distortion_level,
            "source_file": log_filename,
            "": ""  # Empty field to match schema
        },
        "collection": collection_path
    }
    
    # Write labels file
    labels_data = {"events": [event]}
    with open(labels_file, 'w') as f:
        json.dump(labels_data, f, indent=4)
    
    print(f"  Generated labels: {labels_file}")
    return labels_data


def generate_experiment_config(parsed_data: dict, labels_file: str, 
                              log_filename: str, output_file: str, 
                              remote_prefix: str):
    """Generate experiment configuration file
    
    Args:
        parsed_data: Parsed sensor log data
        labels_file: Labels file path
        log_filename: Original log filename
        output_file: Output experiment config file path
        remote_prefix: Remote path prefix for nstrumenta
    """
    
    start_ts = parsed_data["start_ts"]
    stop_ts = parsed_data["stop_ts"]
    duration_s = (stop_ts - start_ts) / 1e6  # Convert microseconds to seconds
    
    experiment_config = {
        "dirname": remote_prefix,
        "labelFiles": [
            {
                "filePath": f"projects/nst-test/data/{remote_prefix}/{os.path.basename(labels_file)}"
            }
        ],
        "description": f"Recorded sensor log: {log_filename}",
        "metadata": {
            "source_type": "recorded_log",
            "source_file": log_filename,
            "total_duration_s": duration_s,
            "start_timestamp": start_ts,
            "end_timestamp": stop_ts,
            **parsed_data["metadata"]
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    print(f"  Generated experiment config: {output_file}")
    return experiment_config


def convert_sensor_log(json_file: str, output_dir: str = ".", 
                      distortion_level: str = "0", upload: bool = True):
    """Convert a single sensor log JSON file to MCAP with labels and experiment config
    
    Args:
        json_file: Path to input JSON sensor log file
        output_dir: Output directory for generated files
        distortion_level: Default distortion level (0=none, 1=low, 2=high)
        upload: Whether to upload files to nstrumenta
    """
    
    # Extract base name from file (remove extension)
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    log_filename = os.path.basename(json_file)
    
    print(f"\nProcessing: {log_filename}")
    
    # Parse the JSON file
    parsed_data = parse_sensor_log_json(json_file)
    
    # Generate output file paths
    mcap_file = os.path.join(output_dir, f"{base_name}.mcap")
    labels_file = os.path.join(output_dir, f"{base_name}.labels.json")
    experiment_file = os.path.join(output_dir, f"{base_name}.experiment.json")
    
    # Convert to MCAP
    convert_to_mcap(parsed_data, mcap_file)
    
    # Generate labels
    remote_prefix = f"recorded_logs/{base_name}"
    collection_path = f"projects/nst-test/data/{remote_prefix}/{os.path.basename(labels_file)}"
    generate_labels(parsed_data, labels_file, log_filename, collection_path, distortion_level)
    
    # Generate experiment config
    generate_experiment_config(parsed_data, labels_file, log_filename, 
                              experiment_file, remote_prefix)
    
    # Upload to nstrumenta if requested
    if upload and nst_client is not None:
        print(f"  Uploading to: {remote_prefix}")
        upload_with_prefix(nst_client, mcap_file, remote_prefix)
        upload_with_prefix(nst_client, labels_file, remote_prefix)
        upload_with_prefix(nst_client, experiment_file, remote_prefix)
    
    print(f"  ✓ Conversion complete!")
    
    return {
        "mcap": mcap_file,
        "labels": labels_file,
        "experiment": experiment_file
    }


def main():
    """Convert all sensor log JSON files in the temp directory"""
    
    # Set up working directory
    working_folder = "./temp"
    reset_to_initial_cwd()
    
    # Find all sensor log JSON files (excluding already-generated label and experiment files)
    json_files = []
    for file in os.listdir(working_folder):
        if (file.startswith("Sensor_Log_") and 
            file.endswith(".json") and 
            not file.endswith(".labels.json") and
            not file.endswith(".experiment.json")):
            json_files.append(os.path.join(working_folder, file))
    
    if not json_files:
        print("No sensor log JSON files found in temp directory")
        return
    
    print(f"Found {len(json_files)} sensor log file(s) to convert")
    
    # Convert each file
    for json_file in json_files:
        try:
            convert_sensor_log(
                json_file, 
                output_dir=working_folder,
                distortion_level="0",  # Default to no distortion
                upload=True
            )
        except Exception as e:
            print(f"  ✗ Error converting {json_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Conversion Summary")
    print("="*60)
    print(f"Total files processed: {len(json_files)}")
    print("\nGenerated files for each log:")
    print("  - .mcap: MCAP sensor data file")
    print("  - .labels.json: Event labels with single event for full recording")
    print("  - .experiment.json: Experiment configuration")
    print("\nFiles uploaded to: projects/nst-test/data/recorded_logs/")
    print("\nThese files can now be used just like synthetic data for:")
    print("  - Training classifiers")
    print("  - Testing algorithms")
    print("  - Data analysis and visualization")


if __name__ == "__main__":
    main()
