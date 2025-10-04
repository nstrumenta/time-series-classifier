# Sensor Log Conversion - Summary

## What Was Created

We've enhanced the time-series-classifier project with the ability to convert recorded sensor log JSON files into MCAP format, making them compatible with the existing synthetic data workflow.

## New Files

### 1. `scripts/convert_sensor_logs.py`
Main conversion script that:
- Parses iPhone sensor log JSON files (or similar formats)
- Converts sensor data to MCAP format with proper channels
- Generates label files with single event spanning the recording
- Creates experiment configuration files
- Uploads all files to Nstrumenta

**Supported sensor types:**
- MAG_AUTOCAL (magnetometer)
- ACCEL_AUTOCAL (accelerometer)  
- GYRO_AUTOCAL (gyroscope)
- Q_MAG_ACCEL (mag+accel quaternion)
- Q_9AXIS (9-axis quaternion)
- LINEAR_ACCEL (linear acceleration)
- TEMPERATURE (temperature sensor)

### 2. `docs/converting_sensor_logs.md`
Comprehensive documentation including:
- Input format specification
- Usage examples (CLI and programmatic)
- Output file descriptions
- Workflow comparison with synthetic data
- Troubleshooting guide

### 3. `examples/using_recorded_data.py`
Example script demonstrating how to:
- Load converted MCAP files
- Read labels and experiment configs
- Display sensor log summaries
- Use data for classification/training

## Modified Files

### `scripts/script_utils.py`
Enhanced `upload_with_prefix()` function to:
- Accept optional `remote_filename` parameter
- Default to basename of local file for cleaner remote paths
- Maintain backward compatibility

## How It Works

### Input Format
```json
{
  "firmware": "FORT (848)",
  "serialNumber": "1073360",
  "deviceModel": "iPhone 15 Pro Max",
  "data": [
    {"ts": 1526934747, "type": "START", "values": []},
    {"ts": 1527031722, "type": "MAG_AUTOCAL", "values": [14.43, 10.73, -44.07]},
    ...
  ]
}
```

### Output Files (per log)
1. **`*.mcap`** - MCAP sensor data with channels
2. **`*.labels.json`** - Event labels (single event for full recording)
3. **`*.experiment.json`** - Experiment configuration

### Label File Schema
```json
{
  "events": [
    {
      "id": "uuid",
      "startTime": {"sec": 1526, "nsec": 934747000},
      "endTime": {"sec": 1562, "nsec": 681897000},
      "metadata": {
        "mag_distortion": "0",
        "source_file": "Sensor_Log_xyz.json",
        "": ""
      },
      "collection": "projects/nst-test/data/recorded_logs/..."
    }
  ]
}
```

## Usage Example

### Convert All Logs
```bash
python scripts/convert_sensor_logs.py
```

### Use Converted Data
```python
# Load and inspect
python examples/using_recorded_data.py

# Use for classification
from scripts.classify import classify_data
classify_data("MAG_DIST_01", "temp/Sensor_Log_xyz.mcap")

# Use for training
from scripts.fine_tune import fine_tune_model
fine_tune_model(
    "MAG_DIST_01",
    ["projects/nst-test/data/recorded_logs/.../xyz.labels.json"],
    "MAG_DIST_RECORDED"
)
```

## Benefits

1. **Unified Workflow**: Recorded data uses same format as synthetic data
2. **Easy Integration**: Compatible with existing training and classification scripts
3. **Metadata Preservation**: All device info preserved in experiment configs
4. **Flexible Labeling**: Initial label can be refined with annotation tools
5. **Cloud Storage**: Automatic upload to Nstrumenta for team access

## Testing Results

Successfully converted two sensor logs:
- `Sensor_Log_sn3360_2025-10-03_01_21_10.json` (35.75 seconds, 6640 messages)
- `Sensor_Log_sn3360_2025-10-03_01_22_48.json` (similar format)

Both converted cleanly with:
- ✓ MCAP files with 7 sensor channels
- ✓ Label files with proper time ranges
- ✓ Experiment configs with device metadata
- ✓ Successful upload to Nstrumenta

## Next Steps

Users can now:
1. **Convert** their recorded sensor logs
2. **Refine** labels by editing the JSON or using annotation tools
3. **Train** models on real-world data
4. **Mix** synthetic and recorded data for robust training
5. **Classify** new recordings using trained models

## Documentation

- Main guide: [`docs/converting_sensor_logs.md`](converting_sensor_logs.md)
- Example usage: [`examples/using_recorded_data.py`](../examples/using_recorded_data.py)
- Updated README: [`README.md`](../README.md)
