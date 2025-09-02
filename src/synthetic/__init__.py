"""
Synthetic data generation module for time series classifier
"""

from .generator import SyntheticDataGenerator
from .math_utils import Vector3, Quaternion, Matrix3x3, SensorCalibration, Pose
from .math_utils import (
    deg2rad, 
    euler_angles_to_quaternion, 
    multiply_quaternions, 
    quaternion_inverse, 
    rotate_vector,
    determinant,
    inverse,
    apply_inverse_sensor_calibration
)

__all__ = [
    'SyntheticDataGenerator',
    'Vector3', 'Quaternion', 'Matrix3x3', 'SensorCalibration', 'Pose',
    'deg2rad', 'euler_angles_to_quaternion', 'multiply_quaternions',
    'quaternion_inverse', 'rotate_vector', 'determinant', 'inverse',
    'apply_inverse_sensor_calibration'
]
