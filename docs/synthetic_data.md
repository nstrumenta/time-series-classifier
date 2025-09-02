# Synthetic Data Generation Module

This module provides a complete Python implementation of synthetic sensor data generation, originally ported from C++. It generates realistic IMU (Inertial Measurement Unit) sensor data including magnetometer, accelerometer, and gyroscope readings based on configurable motion plans.

## Directory Structure

```
src/
├── synthetic/                 # Synthetic data generation module
│   ├── __init__.py           # Main module exports
│   ├── generator.py          # Core data generator class
│   └── math_utils.py         # Mathematical utilities (quaternions, vectors, etc.)
├── mcap_utils/               # MCAP data processing utilities
│   ├── __init__.py           # Utility exports
│   ├── reader.py             # MCAP reading functions
│   ├── visualization.py     # Data plotting and visualization
│   ├── dataset.py            # Dataset creation for ML
│   └── spectrogram.py        # Spectrogram processing
config/                       # Configuration files
├── default_plan.json         # Default motion plan configuration
scripts/                      # Command-line utilities
├── generate_synthetic.py     # CLI for synthetic data generation
examples/                     # Usage examples
├── basic_example.py          # Basic usage demonstration
tests/                        # Unit tests
├── test_synthetic.py         # Comprehensive test suite
```

## Quick Start

### 1. Generate Synthetic Data

```python
from synthetic import SyntheticDataGenerator

generator = SyntheticDataGenerator()
generator.generate("config/default_plan.json", "output.mcap")
```

Or use the command-line interface:

```bash
python scripts/generate_synthetic.py --plan config/default_plan.json --output data.mcap
```

### 2. Read and Analyze Data

```python
from mcap_utils import read_synthetic_sensor_data, plot_synthetic_sensor_data

# Read sensor data
data = read_synthetic_sensor_data("data.mcap")
print(f"Magnetometer samples: {len(data['mag_truth'])}")

# Plot the data
plot_synthetic_sensor_data("data.mcap", channels=["mag_truth", "acc_truth", "gyro_truth"])
```

### 3. Extract Windows for Machine Learning

```python
from mcap_utils import extract_imu_windows

# Extract 1-second windows with 0.5-second overlap
windows = extract_imu_windows("data.mcap", window_size_ns=1e9, step_size_ns=0.5e9)
print(f"Extracted {len(windows)} windows")
```

## Features

### Synthetic Data Generation

- **Realistic Sensor Simulation**: Generates magnetometer, accelerometer, and gyroscope data based on physics
- **Configurable Motion Plans**: Define complex motion sequences with rotations and timing
- **Sensor Calibration**: Apply realistic sensor calibration matrices and bias
- **High-Performance**: Pure Python implementation with minimal dependencies

### Data Processing

- **MCAP Integration**: Full support for MCAP format reading and writing
- **Windowed Extraction**: Extract time-windowed data for machine learning applications
- **Visualization**: Built-in plotting capabilities for sensor data analysis
- **Dataset Creation**: Convert MCAP data to HuggingFace datasets for ML training

## Configuration

The motion plan is defined in a JSON file with the following structure:

```json
{
  "initialization": {
    "pose": {
      "origin": {"lat": 38.446, "lng": -122.687, "height": 0.0},
      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
    },
    "start_time_ns": 0,
    "sample_rate": 30,
    "mag": {"calibration": {...}},
    "acc": {"calibration": {...}},
    "gyro": {"calibration": {...}}
  },
  "segments": [
    {
      "name": "rotation_segment",
      "duration_s": 10.0,
      "rotation_rpy_degrees": {"roll": 45.0, "pitch": 0.0, "yaw": 90.0}
    }
  ]
}
```

## Mathematical Foundation

The synthetic data generator implements:

- **Quaternion Mathematics**: For accurate 3D rotations
- **Sensor Modeling**: Realistic magnetometer, accelerometer, and gyroscope behavior
- **Calibration Effects**: Inverse sensor calibration to simulate raw sensor readings
- **Reference Frames**: Proper handling of sensor coordinate systems

## Testing

Run the comprehensive test suite:

```bash
python -m unittest tests.test_synthetic
```

## Examples

See `examples/basic_example.py` for a complete workflow demonstration.

## Dependencies

- `mcap`: MCAP format support
- `numpy`: Numerical computations
- `matplotlib`: Data visualization (optional)
- `torch`/`torchaudio`: For spectrogram processing (optional)

## Performance

The Python implementation generates synthetic data at similar rates to the original C++ version while providing better integration with the Python ML ecosystem.
