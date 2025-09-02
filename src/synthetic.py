#!/usr/bin/env python3
"""
Python implementation of the synthetic data generator from synthetic.cpp
Generates synthetic sensor data (magnetometer, accelerometer, gyroscope) and pose truth data.
"""

import argparse
import json
import math
import os
from typing import Dict, List, Tuple, Union
import numpy as np
from mcap.writer import Writer


class Vector3:
    """3D vector representation"""
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]


class Quaternion:
    """Quaternion representation for rotations"""
    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def to_list(self) -> List[float]:
        return [self.w, self.x, self.y, self.z]


class Matrix3x3:
    """3x3 matrix representation"""
    def __init__(self, matrix: List[List[float]] = None):
        if matrix is None:
            # Identity matrix
            self.m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        else:
            self.m = matrix


class SensorCalibration:
    """Sensor calibration data structure"""
    def __init__(self, bias: Vector3, matrix: Matrix3x3):
        self.bias = bias
        self.matrix = matrix


class Pose:
    """Pose representation with origin, position, and rotation"""
    def __init__(self, origin: Vector3, position: Vector3, rotation: Quaternion):
        self.origin = origin
        self.position = position
        self.rotation = rotation


def deg2rad(degrees: float) -> float:
    """Convert degrees to radians"""
    return degrees * math.pi / 180.0


def determinant(matrix: Matrix3x3) -> float:
    """Compute the determinant of a 3x3 matrix"""
    m = matrix.m
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
            m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
            m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def inverse(matrix: Matrix3x3) -> Matrix3x3:
    """Compute the inverse of a 3x3 matrix"""
    det = determinant(matrix)
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted")
    
    inv_det = 1.0 / det
    m = matrix.m
    
    inv_matrix = Matrix3x3()
    inv_matrix.m[0][0] = inv_det * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    inv_matrix.m[0][1] = inv_det * (m[0][2] * m[2][1] - m[0][1] * m[2][2])
    inv_matrix.m[0][2] = inv_det * (m[0][1] * m[1][2] - m[0][2] * m[1][1])
    inv_matrix.m[1][0] = inv_det * (m[1][2] * m[2][0] - m[1][0] * m[2][2])
    inv_matrix.m[1][1] = inv_det * (m[0][0] * m[2][2] - m[0][2] * m[2][0])
    inv_matrix.m[1][2] = inv_det * (m[0][2] * m[1][0] - m[0][0] * m[1][2])
    inv_matrix.m[2][0] = inv_det * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    inv_matrix.m[2][1] = inv_det * (m[0][1] * m[2][0] - m[0][0] * m[2][1])
    inv_matrix.m[2][2] = inv_det * (m[0][0] * m[1][1] - m[0][1] * m[1][0])
    
    return inv_matrix


def apply_inverse_sensor_calibration(calibrated: Vector3, calibration: SensorCalibration) -> Vector3:
    """Apply the inverse sensor calibration"""
    inv_matrix = inverse(calibration.matrix)
    
    raw = Vector3()
    raw.x = (inv_matrix.m[0][0] * calibrated.x + inv_matrix.m[0][1] * calibrated.y +
             inv_matrix.m[0][2] * calibrated.z + calibration.bias.x)
    raw.y = (inv_matrix.m[1][0] * calibrated.x + inv_matrix.m[1][1] * calibrated.y +
             inv_matrix.m[1][2] * calibrated.z + calibration.bias.y)
    raw.z = (inv_matrix.m[2][0] * calibrated.x + inv_matrix.m[2][1] * calibrated.y +
             inv_matrix.m[2][2] * calibrated.z + calibration.bias.z)
    
    return raw


def euler_angles_to_quaternion(euler: Vector3) -> Quaternion:
    """Convert Euler angles (roll, pitch, yaw) to quaternion"""
    roll, pitch, yaw = euler.x, euler.y, euler.z
    
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    
    return q


def multiply_quaternions(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """Multiply two quaternions"""
    result = Quaternion()
    result.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    result.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y
    result.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x
    result.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
    return result


def quaternion_inverse(q: Quaternion) -> Quaternion:
    """Compute the inverse (conjugate) of a quaternion"""
    return Quaternion(q.w, -q.x, -q.y, -q.z)


def rotate_vector(vector: Vector3, quaternion: Quaternion) -> Vector3:
    """Rotate a vector by a quaternion"""
    # Convert vector to quaternion
    v_quat = Quaternion(0, vector.x, vector.y, vector.z)
    
    # Compute q * v * q_inverse
    q_inv = quaternion_inverse(quaternion)
    result_quat = multiply_quaternions(multiply_quaternions(quaternion, v_quat), q_inv)
    
    return Vector3(result_quat.x, result_quat.y, result_quat.z)


def write_message(writer: Writer, channel_ids: Dict[str, int], channel_name: str, 
                 values: List[float], current_time: int):
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


def load_plan(plan_file: str) -> dict:
    """Load the plan JSON file"""
    with open(plan_file, 'r') as f:
        return json.load(f)


def create_sensor_calibration(calibration_data: dict) -> SensorCalibration:
    """Create a SensorCalibration object from JSON data"""
    bias = Vector3(
        calibration_data["bias"]["x"],
        calibration_data["bias"]["y"],
        calibration_data["bias"]["z"]
    )
    matrix = Matrix3x3(calibration_data["matrix"])
    return SensorCalibration(bias, matrix)


def generate_synthetic_data(plan_file: str, output_file: str):
    """Main function to generate synthetic data"""
    
    # Load plan
    plan = load_plan(plan_file)
    
    # Set up MCAP writer
    with open(output_file, "wb") as f:
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
        
        # Register channels
        channel_names = ["pose_truth", "mag_truth", "acc_truth", "gyro_truth",
                        "pose", "mag_raw", "acc_raw", "gyro_raw"]
        channel_ids = {}
        
        for name in channel_names:
            channel_id = writer.register_channel(
                topic=name,
                message_encoding="json",
                schema_id=schema_id
            )
            channel_ids[name] = channel_id
        
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
        mag_calibration = create_sensor_calibration(init["mag"]["calibration"])
        acc_calibration = create_sensor_calibration(init["acc"]["calibration"])
        gyro_calibration = create_sensor_calibration(init["gyro"]["calibration"])
        
        # Initialize timing
        current_time = int(init["start_time_ns"])
        sample_rate = init["sample_rate"]
        sample_interval = int(1e9 / sample_rate)
        dt = 1.0 / sample_rate
        
        # Reference vectors
        accel_ref = Vector3(0.0, 0.0, 1.0)
        mag_ref = Vector3(0.0, 1.0, 0.0)  # TODO: use inclination and declination
        
        print(f"Plan file: {plan_file}")
        
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
                
                # Magnetometer
                mag_truth = rotate_vector(mag_ref, inverse_pose_rotation)
                write_message(writer, channel_ids, "mag_truth", mag_truth.to_list(), current_time)
                mag_raw = apply_inverse_sensor_calibration(mag_truth, mag_calibration)
                write_message(writer, channel_ids, "mag_raw", mag_raw.to_list(), current_time)
                
                # Accelerometer
                acc_truth = rotate_vector(accel_ref, inverse_pose_rotation)
                write_message(writer, channel_ids, "acc_truth", acc_truth.to_list(), current_time)
                acc_raw = apply_inverse_sensor_calibration(acc_truth, acc_calibration)
                write_message(writer, channel_ids, "acc_raw", acc_raw.to_list(), current_time)
                
                # Gyroscope
                gyro_truth = Vector3(
                    2.0 * segment_rotation.x / dt,
                    2.0 * segment_rotation.y / dt,
                    2.0 * segment_rotation.z / dt
                )
                write_message(writer, channel_ids, "gyro_truth", gyro_truth.to_list(), current_time)
                gyro_raw = apply_inverse_sensor_calibration(gyro_truth, gyro_calibration)
                write_message(writer, channel_ids, "gyro_raw", gyro_raw.to_list(), current_time)
                
                # Pose truth
                pose_truth_values = [
                    pose.origin.x, pose.origin.y, pose.origin.z,  # lat, lng, height
                    pose.position.x, pose.position.y, pose.position.z,  # x, y, z
                    pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z  # quat
                ]
                write_message(writer, channel_ids, "pose_truth", pose_truth_values, current_time)
                
                current_time += sample_interval
        
        writer.finish()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate synthetic sensor data")
    parser.add_argument("-p", "--plan", required=True, help="Plan file path")
    parser.add_argument("-o", "--outfile", required=True, help="Output MCAP file path")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.plan):
        print(f"Error: Plan file '{args.plan}' does not exist")
        return 1
    
    try:
        generate_synthetic_data(args.plan, args.outfile)
        print(f"Successfully generated synthetic data: {args.outfile}")
    except Exception as e:
        print(f"Error generating synthetic data: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
