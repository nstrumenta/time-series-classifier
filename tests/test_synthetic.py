#!/usr/bin/env python3
"""
Tests for the synthetic data generation module
"""

import unittest
import tempfile
import os
import sys
import json

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from synthetic import SyntheticDataGenerator, Vector3, Quaternion, Matrix3x3
from synthetic.math_utils import deg2rad, euler_angles_to_quaternion, multiply_quaternions
from mcap_utils import read_synthetic_sensor_data


class TestMathUtils(unittest.TestCase):
    """Test mathematical utility functions"""
    
    def test_vector3(self):
        """Test Vector3 class"""
        v = Vector3(1.0, 2.0, 3.0)
        self.assertEqual(v.to_list(), [1.0, 2.0, 3.0])
    
    def test_quaternion(self):
        """Test Quaternion class"""
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.assertEqual(q.to_list(), [1.0, 0.0, 0.0, 0.0])
    
    def test_deg2rad(self):
        """Test degree to radian conversion"""
        self.assertAlmostEqual(deg2rad(180), 3.14159265359, places=5)
        self.assertAlmostEqual(deg2rad(90), 1.57079632679, places=5)
    
    def test_euler_to_quaternion(self):
        """Test Euler angle to quaternion conversion"""
        euler = Vector3(0, 0, 0)  # Identity rotation
        q = euler_angles_to_quaternion(euler)
        self.assertAlmostEqual(q.w, 1.0, places=5)
        self.assertAlmostEqual(q.x, 0.0, places=5)
        self.assertAlmostEqual(q.y, 0.0, places=5)
        self.assertAlmostEqual(q.z, 0.0, places=5)


class TestSyntheticDataGenerator(unittest.TestCase):
    """Test synthetic data generator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = SyntheticDataGenerator()
        
        # Create a minimal test plan
        self.test_plan = {
            "initialization": {
                "pose": {
                    "origin": {"lat": 0.0, "lng": 0.0, "height": 0.0},
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
                },
                "start_time_ns": 0,
                "sample_rate": 10,
                "mag": {
                    "calibration": {
                        "bias": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                    }
                },
                "acc": {
                    "calibration": {
                        "bias": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                    }
                },
                "gyro": {
                    "calibration": {
                        "bias": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                    }
                }
            },
            "segments": [
                {
                    "name": "test_segment",
                    "duration_s": 1.0,
                    "rotation_rpy_degrees": {"roll": 10.0, "pitch": 0.0, "yaw": 0.0}
                }
            ]
        }
    
    def test_load_plan(self):
        """Test plan loading"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_plan, f)
            plan_file = f.name
        
        try:
            plan = self.generator.load_plan(plan_file)
            self.assertEqual(plan["initialization"]["sample_rate"], 10)
            self.assertEqual(len(plan["segments"]), 1)
        finally:
            os.unlink(plan_file)
    
    def test_generate_synthetic_data(self):
        """Test complete synthetic data generation"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as plan_f:
            json.dump(self.test_plan, plan_f)
            plan_file = plan_f.name
        
        with tempfile.NamedTemporaryFile(suffix='.mcap', delete=False) as mcap_f:
            output_file = mcap_f.name
        
        try:
            # Generate data
            self.generator.generate(plan_file, output_file, verbose=False)
            
            # Verify file exists and has content
            self.assertTrue(os.path.exists(output_file))
            self.assertGreater(os.path.getsize(output_file), 0)
            
            # Read and verify data
            data = read_synthetic_sensor_data(output_file)
            
            # Check that we have the expected channels
            expected_channels = ["mag_truth", "acc_truth", "gyro_truth", "mag_raw", "acc_raw", "gyro_raw", "pose_truth"]
            for channel in expected_channels:
                self.assertIn(channel, data)
                self.assertGreater(len(data[channel]), 0)
            
            # Check sample rate (should be ~10 Hz for 1 second = ~10 samples)
            mag_data = data["mag_truth"]
            self.assertGreaterEqual(len(mag_data), 8)  # Allow some tolerance
            self.assertLessEqual(len(mag_data), 12)
            
        finally:
            if os.path.exists(plan_file):
                os.unlink(plan_file)
            if os.path.exists(output_file):
                os.unlink(output_file)


class TestMCAPUtils(unittest.TestCase):
    """Test MCAP utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Generate a small test file
        self.generator = SyntheticDataGenerator()
        
        self.test_plan = {
            "initialization": {
                "pose": {
                    "origin": {"lat": 0.0, "lng": 0.0, "height": 0.0},
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
                },
                "start_time_ns": 0,
                "sample_rate": 10,
                "mag": {
                    "calibration": {
                        "bias": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                    }
                },
                "acc": {
                    "calibration": {
                        "bias": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                    }
                },
                "gyro": {
                    "calibration": {
                        "bias": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                    }
                }
            },
            "segments": [
                {
                    "name": "test_segment",
                    "duration_s": 2.0,
                    "rotation_rpy_degrees": {"roll": 10.0, "pitch": 0.0, "yaw": 0.0}
                }
            ]
        }
        
        # Create test files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as plan_f:
            json.dump(self.test_plan, plan_f)
            self.plan_file = plan_f.name
        
        with tempfile.NamedTemporaryFile(suffix='.mcap', delete=False) as mcap_f:
            self.mcap_file = mcap_f.name
        
        # Generate test data
        self.generator.generate(self.plan_file, self.mcap_file, verbose=False)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.plan_file):
            os.unlink(self.plan_file)
        if os.path.exists(self.mcap_file):
            os.unlink(self.mcap_file)
    
    def test_read_synthetic_sensor_data(self):
        """Test reading synthetic sensor data"""
        data = read_synthetic_sensor_data(self.mcap_file)
        
        # Check that we have data for all channels
        expected_channels = ["mag_truth", "acc_truth", "gyro_truth", "mag_raw", "acc_raw", "gyro_raw", "pose_truth"]
        for channel in expected_channels:
            self.assertIn(channel, data)
            self.assertGreater(len(data[channel]), 0)
        
        # Check data format
        mag_data = data["mag_truth"]
        timestamp, values = mag_data[0]
        self.assertIsInstance(timestamp, int)
        self.assertIsInstance(values, list)
        self.assertEqual(len(values), 3)  # x, y, z


if __name__ == '__main__':
    unittest.main()
