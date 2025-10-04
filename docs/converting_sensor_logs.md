# Converting Recorded Sensor Logs to MCAP

This document describes how to convert recorded sensor log JSON files to MCAP format for use in the time-series classifier pipeline.

## Overview

The `convert_sensor_logs.py` script converts sensor log JSON files (like those from iPhone recordings) into MCAP format with labels and experiment configurations, making them compatible with the same workflow as synthetic data.

## Input Format

The script expects JSON files with the following structure:

```json
{
  "firmware": "FORT (848)",
  "serialNumber": "1073360",
  "appVersion": "1.0.0 (196)",
  "deviceModel": "iPhone 15 Pro Max",
  "deviceName": "iPhone",
  "osVersion": "Version 18.6.2 (Build 22G100)",
  "userLabel": "",
  "data": [
    {"ts": 1526934747, "type": "START", "values": []},
    {"ts": 1527031722, "type": "MAG_AUTOCAL", "values": [14.43116, 10.73493, -44.07435]},
    {"ts": 1527031630, "type": "ACCEL_AUTOCAL", "values": [-0.1543563, 0.08429588, -0.9823679]},
    {"ts": 1527034529, "type": "GYRO_AUTOCAL", "values": [0.001911115, 0.00001167506, -0.001957926]},
    ...
    {"ts": 1562681897, "type": "STOP", "values": []}
  ]
}
```

### Supported Sensor Types

The script converts the following sensor types to MCAP channels:

- `MAG_AUTOCAL` → `mag_raw` (magnetometer)
- `ACCEL_AUTOCAL` → `acc_raw` (accelerometer)
- `GYRO_AUTOCAL` → `gyro_raw` (gyroscope)
- `Q_MAG_ACCEL` → `q_mag_accel` (quaternion from mag+accel)
- `Q_9AXIS` → `q_9axis` (9-axis quaternion)
- `LINEAR_ACCEL` → `linear_accel` (linear acceleration)
- `TEMPERATURE` → `temperature`

### Timestamp Handling

- Input timestamps are in **microseconds**
- Converted to **nanoseconds** for MCAP compatibility
- Time range determined by `START`/`STOP` events or first/last data points

## Usage

### Basic Usage

Convert all sensor log JSON files in the `temp/` directory:

```bash
cd /workspaces/time-series-classifier
python scripts/convert_sensor_logs.py
```

### Programmatic Usage

```python
from convert_sensor_logs import convert_sensor_log

# Convert a single file
convert_sensor_log(
    json_file="temp/Sensor_Log_sn3360_2025-10-03_01_21_10.json",
    output_dir="temp",
    distortion_level="0",  # 0=none, 1=low, 2=high
    upload=True
)
```

### Custom Distortion Levels

You can specify different magnetic distortion levels when converting:

```python
# No distortion (default)
convert_sensor_log(json_file, distortion_level="0")

# Low distortion
convert_sensor_log(json_file, distortion_level="1")

# High distortion  
convert_sensor_log(json_file, distortion_level="2")
```

## Output Files

For each input file like `Sensor_Log_sn3360_2025-10-03_01_21_10.json`, the script generates:

### 1. MCAP File (`*.mcap`)
Binary MCAP file containing sensor data with proper channels and timestamps.

### 2. Labels File (`*.labels.json`)
Event labels in the standard format:

```json
{
  "events": [
    {
      "id": "uuid-here",
      "startTime": {
        "sec": 1526,
        "nsec": 934747000
      },
      "endTime": {
        "sec": 1562,
        "nsec": 681897000
      },
      "metadata": {
        "mag_distortion": "0",
        "source_file": "Sensor_Log_sn3360_2025-10-03_01_21_10.json",
        "": ""
      },
      "collection": "projects/nst-test/data/recorded_logs/Sensor_Log_sn3360_2025-10-03_01_21_10/Sensor_Log_sn3360_2025-10-03_01_21_10.labels.json"
    }
  ]
}
```

**Note:** The initial conversion creates a **single event** spanning the entire recording. You can manually edit this file or use annotation tools to add more specific event labels.

### 3. Experiment Configuration (`*.experiment.json`)
Experiment config linking data and labels:

```json
{
  "dirname": "recorded_logs/Sensor_Log_sn3360_2025-10-03_01_21_10",
  "labelFiles": [
    {
      "filePath": "projects/nst-test/data/recorded_logs/Sensor_Log_sn3360_2025-10-03_01_21_10/Sensor_Log_sn3360_2025-10-03_01_21_10.labels.json"
    }
  ],
  "description": "Recorded sensor log: Sensor_Log_sn3360_2025-10-03_01_21_10.json",
  "metadata": {
    "source_type": "recorded_log",
    "source_file": "Sensor_Log_sn3360_2025-10-03_01_21_10.json",
    "total_duration_s": 35747.15,
    "start_timestamp": 1526934747,
    "end_timestamp": 1562681897,
    "firmware": "FORT (848)",
    "serialNumber": "1073360",
    "appVersion": "1.0.0 (196)",
    "deviceModel": "iPhone 15 Pro Max",
    "deviceName": "iPhone",
    "osVersion": "Version 18.6.2 (Build 22G100)",
    "userLabel": ""
  }
}
```

## File Organization

Converted files are uploaded to Nstrumenta with the following structure:

```
projects/nst-test/data/recorded_logs/
├── Sensor_Log_sn3360_2025-10-03_01_21_10/
│   ├── Sensor_Log_sn3360_2025-10-03_01_21_10.mcap
│   ├── Sensor_Log_sn3360_2025-10-03_01_21_10.labels.json
│   └── Sensor_Log_sn3360_2025-10-03_01_21_10.experiment.json
└── Sensor_Log_sn3360_2025-10-03_01_22_48/
    ├── Sensor_Log_sn3360_2025-10-03_01_22_48.mcap
    ├── Sensor_Log_sn3360_2025-10-03_01_22_48.labels.json
    └── Sensor_Log_sn3360_2025-10-03_01_22_48.experiment.json
```

## Using Converted Data

Once converted, recorded sensor logs can be used exactly like synthetic data:

### For Training

```python
from scripts.fine_tune import fine_tune_model

fine_tune_model(
    base_model_id="MAG_DIST_01",
    label_files=[
        "projects/nst-test/data/recorded_logs/Sensor_Log_sn3360_2025-10-03_01_21_10/Sensor_Log_sn3360_2025-10-03_01_21_10.labels.json"
    ],
    output_model_id="MAG_DIST_RECORDED"
)
```

### For Classification

```python
from scripts.classify import classify_data

classify_data(
    model_id="MAG_DIST_01",
    mcap_file="temp/Sensor_Log_sn3360_2025-10-03_01_21_10.mcap"
)
```

### In Notebooks

```python
# Load the MCAP file
from mcap_utils.reader import McapReader

reader = McapReader("temp/Sensor_Log_sn3360_2025-10-03_01_21_10.mcap")
data = reader.read()

# Load the labels
import json
with open("temp/Sensor_Log_sn3360_2025-10-03_01_21_10.labels.json") as f:
    labels = json.load(f)
```

## Workflow Comparison

### Synthetic Data
```
Plan JSON → Generate → MCAP + Labels + Experiment Config
```

### Recorded Data
```
Sensor Log JSON → Convert → MCAP + Labels + Experiment Config
```

Both paths produce the same output format, enabling unified processing!

## Tips

1. **Label Refinement**: The initial conversion creates one label for the entire recording. Use annotation tools or manually edit the labels file to add more specific event boundaries.

2. **Batch Processing**: Place all sensor log JSON files in `temp/` and run the script once to convert all of them.

3. **Distortion Classification**: Start with distortion level "0" (none) and update labels based on analysis or manual review.

4. **Metadata Preservation**: All device metadata from the original JSON is preserved in the experiment config for reference.

5. **Time Alignment**: Ensure timestamps are in microseconds in the input JSON for proper conversion.

## Troubleshooting

**Q: Script fails with "No sensor log JSON files found"**  
A: Ensure files start with `Sensor_Log_` and end with `.json`, and are in the `temp/` directory.

**Q: Missing channels in MCAP file**  
A: Check that your input JSON contains the expected sensor types. The script only converts sensor types it recognizes.

**Q: Upload fails**  
A: Verify your `NST_API_KEY` environment variable is set and you have network connectivity.

**Q: Timestamps seem wrong**  
A: Input timestamps should be in microseconds. Check the original JSON file format.
