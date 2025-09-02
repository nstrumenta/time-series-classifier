"""
Mathematical utilities for synthetic data generation
Contains quaternion math, vector operations, and sensor calibration functions
"""

import math
from typing import List


class Vector3:
    """3D vector representation"""
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]

    def __repr__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"


class Quaternion:
    """Quaternion representation for rotations"""
    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def to_list(self) -> List[float]:
        return [self.w, self.x, self.y, self.z]

    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"


class Matrix3x3:
    """3x3 matrix representation"""
    def __init__(self, matrix: List[List[float]] = None):
        if matrix is None:
            # Identity matrix
            self.m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        else:
            self.m = matrix

    def __repr__(self):
        return f"Matrix3x3({self.m})"


class SensorCalibration:
    """Sensor calibration data structure"""
    def __init__(self, bias: Vector3, matrix: Matrix3x3):
        self.bias = bias
        self.matrix = matrix

    def __repr__(self):
        return f"SensorCalibration(bias={self.bias}, matrix={self.matrix})"


class Pose:
    """Pose representation with origin, position, and rotation"""
    def __init__(self, origin: Vector3, position: Vector3, rotation: Quaternion):
        self.origin = origin
        self.position = position
        self.rotation = rotation

    def __repr__(self):
        return f"Pose(origin={self.origin}, position={self.position}, rotation={self.rotation})"


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
