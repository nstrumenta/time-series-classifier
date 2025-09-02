"""
Synthetic data generator for sensor data simulation
"""

import json
import os
from typing import Dict, List
from mcap.writer import Writer

from .math_utils import (
    Vector3, Quaternion, Matrix3x3, SensorCalibration, Pose,
    deg2rad, euler_angles_to_quaternion, multiply_quaternions,
    quaternion_inverse, rotate_vector, apply_inverse_sensor_calibration
)


class SyntheticDataGenerator:
    """Generate synthetic sensor data based on motion plans"""
    
    def __init__(self):
        self.channel_names = [
            "pose_truth", "mag_truth", "acc_truth", "gyro_truth",
            "pose", "mag_raw", "acc_raw", "gyro_raw"
        ]
        
    def load_plan(self, plan_file: str) -> dict:
        """Load the plan JSON file"""
        if not os.path.exists(plan_file):
            raise FileNotFoundError(f"Plan file '{plan_file}' not found")
            
        with open(plan_file, 'r') as f:
            return json.load(f)
    
    def create_sensor_calibration(self, calibration_data: dict) -> SensorCalibration:
        """Create a SensorCalibration object from JSON data"""
        bias = Vector3(
            calibration_data["bias"]["x"],
            calibration_data["bias"]["y"],
            calibration_data["bias"]["z"]
        )
        matrix = Matrix3x3(calibration_data["matrix"])
        return SensorCalibration(bias, matrix)
    
    def write_message(self, writer: Writer, channel_ids: Dict[str, int], 
                     channel_name: str, values: List[float], current_time: int):
        """Write a message to the MCAP file"""
        data = {
            "id": channel_name,
            "timestamp": current_time,
            "values": values
        }
        serialized = json.dumps(data)
        
        writer.add_message(
            channel_ids[channel_name],
            log_time=current_time,
            publish_time=current_time,
            data=serialized.encode()
        )
    
    def setup_mcap_writer(self, writer: Writer) -> Dict[str, int]:
        """Setup MCAP writer with schema and channels"""
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
        
        # Register channels
        channel_ids = {}
        for name in self.channel_names:
            channel_id = writer.register_channel(
                topic=name,
                message_encoding="json",
                schema_id=schema_id
            )
            channel_ids[name] = channel_id
        
        return channel_ids
    
    def generate_labels(self, plan_data: dict, labels_file: str, collection_path: str = None):
        """Generate label file based on plan segments with magnetic distortion classification
        
        Args:
            plan_data: Plan data dictionary containing segments with mag_distortion info
            labels_file: Output labels JSON file path  
            collection_path: Path for the collection field in labels
        """
        import uuid
        
        if collection_path is None:
            collection_path = f"projects/nst-test/data/{os.path.splitext(os.path.basename(labels_file))[0]}/{os.path.basename(labels_file)}"
        
        events = []
        current_time_s = 0
        
        for segment in plan_data["segments"]:
            duration_s = segment["duration_s"]
            
            # Check if segment has magnetic distortion info to classify
            if "mag_distortion" in segment:
                distortion_info = segment["mag_distortion"]
                distortion_level = distortion_info.get("level", "none")
                
                # Create label for the entire segment duration
                start_time_ns = int(current_time_s * 1e9)
                end_time_ns = int((current_time_s + duration_s) * 1e9)
                
                # Map distortion level to classification values (matching example_labels format)
                if distortion_level == "none":
                    distortion_value = "0"
                elif distortion_level == "low":
                    distortion_value = "1" 
                elif distortion_level == "high":
                    distortion_value = "2"
                else:
                    distortion_value = "0"  # default to none
                
                event = {
                    "id": str(uuid.uuid4()),
                    "startTime": {
                        "sec": int(start_time_ns // 1e9),
                        "nsec": int(start_time_ns % 1e9)
                    },
                    "endTime": {
                        "sec": int(end_time_ns // 1e9), 
                        "nsec": int(end_time_ns % 1e9)
                    },
                    "metadata": {
                        "mag_distortion": distortion_value,
                        "": ""  # Empty field to match example schema
                    },
                    "collection": collection_path
                }
                
                events.append(event)
            
            current_time_s += duration_s
        
        # Write labels file (matching exact schema format)
        labels_data = {"events": events}
        with open(labels_file, 'w') as f:
            json.dump(labels_data, f, indent=4)  # Use 4-space indent to match example
        
        return labels_data

    def generate(self, plan_file: str = None, output_file: str = None, plan_data: dict = None, verbose: bool = True):
        """Generate synthetic data from plan file or plan data
        
        Args:
            plan_file: Path to JSON plan file (deprecated, use plan_data instead)
            output_file: Output MCAP file path
            plan_data: Plan data as dictionary (preferred method)
            verbose: Whether to print progress information
        """
        
        # Load plan from file or use provided data
        if plan_data is not None:
            plan = plan_data
            if verbose:
                print(f"Using provided plan data")
        elif plan_file is not None:
            plan = self.load_plan(plan_file)
            if verbose:
                print(f"Plan file: {plan_file}")
        else:
            raise ValueError("Either plan_file or plan_data must be provided")
        
        # Set up MCAP writer
        with open(output_file, "wb") as f:
            writer = Writer(f)
            channel_ids = self.setup_mcap_writer(writer)
            writer.start()
            
            # Initialize from plan
            init = plan["initialization"]
            
            # Create pose
            pose = Pose(
                origin=Vector3(
                    init["pose"]["origin"]["lat"],
                    init["pose"]["origin"]["lng"],
                    init["pose"]["origin"]["height"]
                ),
                position=Vector3(
                    init["pose"]["position"]["x"],
                    init["pose"]["position"]["y"],
                    init["pose"]["position"]["z"]
                ),
                rotation=Quaternion(
                    init["pose"]["rotation"]["w"],
                    init["pose"]["rotation"]["x"],
                    init["pose"]["rotation"]["y"],
                    init["pose"]["rotation"]["z"]
                )
            )
            
            # Create sensor calibrations
            mag_calibration = self.create_sensor_calibration(init["mag"]["calibration"])
            acc_calibration = self.create_sensor_calibration(init["acc"]["calibration"])
            gyro_calibration = self.create_sensor_calibration(init["gyro"]["calibration"])
            
            # Initialize timing
            current_time = int(init["start_time_ns"])
            sample_rate = init["sample_rate"]
            sample_interval = int(1e9 / sample_rate)
            dt = 1.0 / sample_rate
            
            # Reference vectors
            accel_ref = Vector3(0.0, 0.0, 1.0)
            mag_ref = Vector3(0.0, 1.0, 0.0)  # TODO: use inclination and declination
            
            # Process segments
            for segment in plan["segments"]:
                name = segment["name"]
                duration_s = segment["duration_s"]
                duration_ns = int(duration_s * 1e9)
                
                samples_in_segment = int(duration_s * sample_rate)
                
                # Initialize segment rotation
                rotation_rpy = segment["rotation_rpy_degrees"]
                segment_rotation = euler_angles_to_quaternion(Vector3(
                    deg2rad(rotation_rpy["roll"] / samples_in_segment),
                    deg2rad(rotation_rpy["pitch"] / samples_in_segment),
                    deg2rad(rotation_rpy["yaw"] / samples_in_segment)
                ))
                
                if verbose:
                    print(f"Segment name: {name}")
                    print(f"Duration: {duration_s}")
                    print(f"Rotation: w={segment_rotation.w}, x={segment_rotation.x}, "
                          f"y={segment_rotation.y}, z={segment_rotation.z}")
                
                segment_end_time = current_time + duration_ns
                
                # Iterate over the duration and add events
                while current_time < segment_end_time:
                    # Update pose rotation
                    pose.rotation = multiply_quaternions(pose.rotation, segment_rotation)
                    inverse_pose_rotation = quaternion_inverse(pose.rotation)
                    
                    # Magnetometer (with optional world-frame distortion)
                    mag_truth = rotate_vector(mag_ref, inverse_pose_rotation)
                    
                    # Apply magnetic distortion if specified in segment
                    if "mag_distortion" in segment:
                        distortion_info = segment["mag_distortion"]
                        distortion_level = distortion_info.get("level", "none")
                        
                        if distortion_level != "none":
                            # World-frame distortion vector
                            world_distortion = Vector3(
                                distortion_info.get("world_x", 0.0),
                                distortion_info.get("world_y", 0.0), 
                                distortion_info.get("world_z", 0.0)
                            )
                            
                            # Apply scaling based on distortion level
                            if distortion_level == "low":
                                scale = 0.2  # 20% of specified distortion
                            elif distortion_level == "high":
                                scale = 1.0  # 100% of specified distortion
                            else:
                                scale = 0.0
                            
                            # Rotate distortion to sensor frame and add to truth
                            sensor_distortion = rotate_vector(world_distortion, inverse_pose_rotation)
                            mag_truth = Vector3(
                                mag_truth.x + sensor_distortion.x * scale,
                                mag_truth.y + sensor_distortion.y * scale,
                                mag_truth.z + sensor_distortion.z * scale
                            )
                    
                    self.write_message(writer, channel_ids, "mag_truth", mag_truth.to_list(), current_time)
                    mag_raw = apply_inverse_sensor_calibration(mag_truth, mag_calibration)
                    self.write_message(writer, channel_ids, "mag_raw", mag_raw.to_list(), current_time)
                    
                    # Accelerometer
                    acc_truth = rotate_vector(accel_ref, inverse_pose_rotation)
                    self.write_message(writer, channel_ids, "acc_truth", acc_truth.to_list(), current_time)
                    acc_raw = apply_inverse_sensor_calibration(acc_truth, acc_calibration)
                    self.write_message(writer, channel_ids, "acc_raw", acc_raw.to_list(), current_time)
                    
                    # Gyroscope
                    gyro_truth = Vector3(
                        2.0 * segment_rotation.x / dt,
                        2.0 * segment_rotation.y / dt,
                        2.0 * segment_rotation.z / dt
                    )
                    self.write_message(writer, channel_ids, "gyro_truth", gyro_truth.to_list(), current_time)
                    gyro_raw = apply_inverse_sensor_calibration(gyro_truth, gyro_calibration)
                    self.write_message(writer, channel_ids, "gyro_raw", gyro_raw.to_list(), current_time)
                    
                    # Pose truth
                    pose_truth_values = [
                        pose.origin.x, pose.origin.y, pose.origin.z,  # lat, lng, height
                        pose.position.x, pose.position.y, pose.position.z,  # x, y, z
                        pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z  # quat
                    ]
                    self.write_message(writer, channel_ids, "pose_truth", pose_truth_values, current_time)
                    
                    current_time += sample_interval
            
            writer.finish()
        
        if verbose:
            file_size = os.path.getsize(output_file)
            print(f"Successfully generated synthetic data: {output_file} ({file_size} bytes)")
    
    def generate_from_plan_data(self, plan_data: dict, output_file: str, labels_file: str = None, verbose: bool = True):
        """Convenience method to generate from plan data dictionary
        
        Args:
            plan_data: Plan data as dictionary
            output_file: Output MCAP file path
            labels_file: Optional output labels JSON file path
            verbose: Whether to print progress information
        """
        # Generate the MCAP data
        self.generate(plan_data=plan_data, output_file=output_file, verbose=verbose)
        
        # Generate labels if requested and plan has label metadata
        if labels_file and self._has_label_metadata(plan_data):
            if verbose:
                print(f"Generating labels: {labels_file}")
            self.generate_labels(plan_data, labels_file)
        
        return output_file
    
    def _has_label_metadata(self, plan_data: dict) -> bool:
        """Check if plan has magnetic distortion metadata in any segment"""
        for segment in plan_data.get("segments", []):
            if "mag_distortion" in segment:
                return True
        return False
