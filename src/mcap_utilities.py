import json
from mcap.reader import make_reader
from mcap.writer import Writer
import matplotlib.pyplot as plt
import numpy as np
import torch, torchaudio
from typing import List
from datasets import Dataset, Array2D, ClassLabel, Features
import io
import base64


# Serialize the numpy array to a base64 string
def serialize_numpy_array(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=True)
    buffer.seek(0)
    array_bytes = buffer.read()
    array_base64 = base64.b64encode(array_bytes).decode("utf-8")
    return array_base64


# Deserialize the base64 string back to a numpy array
def deserialize_numpy_array(array_base64):
    array_bytes = base64.b64decode(array_base64)
    buffer = io.BytesIO(array_bytes)
    array = np.load(buffer, allow_pickle=True)
    return array


def print_mcap_summary(mcap_summary):
    if mcap_summary is None:
        print("No summary available.")
        return

    print("MCAP File Summary:")
    for channel in mcap_summary.channels.items():
        print(f"  - Channel: {channel}")


def read_mcap(file_name):
    with open(file_name, "rb") as f:
        reader = make_reader(f)
        i = 0
        for (
            message
        ) in (
            reader.iter_messages()
        ):  # define specific topics using reader.iter_messages(topics=["/{{topic}}"]) ex. /diagnostics
            json_str = message.data.decode("utf8").replace("'", '"')
            # json_data = json.loads(json_str)
            print(json_str)


def read_synthetic_sensor_data(mcap_file, channels=None):
    """
    Read synthetic sensor data from MCAP file
    
    Args:
        mcap_file: Path to MCAP file
        channels: List of channel names to read (default: all channels)
    
    Returns:
        Dictionary with channel names as keys and lists of (timestamp, values) tuples as values
    """
    if channels is None:
        channels = ["mag_truth", "acc_truth", "gyro_truth", "mag_raw", "acc_raw", "gyro_raw", "pose_truth"]
    
    data = {channel: [] for channel in channels}
    
    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        
        for schema, channel, message in reader.iter_messages():
            if channel.topic in channels:
                json_str = message.data.decode("utf8")
                json_data = json.loads(json_str)
                data[channel.topic].append((json_data["timestamp"], json_data["values"]))
    
    return data


def plot_synthetic_sensor_data(mcap_file, channels=None, time_range=None):
    """
    Plot synthetic sensor data from MCAP file
    
    Args:
        mcap_file: Path to MCAP file
        channels: List of channel names to plot (default: truth channels)
        time_range: Tuple of (start_time, end_time) in nanoseconds (default: all data)
    """
    if channels is None:
        channels = ["mag_truth", "acc_truth", "gyro_truth"]
    
    data = read_synthetic_sensor_data(mcap_file, channels)
    
    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 3 * len(channels)), sharex=True)
    if len(channels) == 1:
        axes = [axes]
    
    for i, channel in enumerate(channels):
        if channel not in data or not data[channel]:
            continue
            
        timestamps = [item[0] for item in data[channel]]
        values = [item[1] for item in data[channel]]
        
        # Filter by time range if specified
        if time_range:
            start_time, end_time = time_range
            filtered_data = [(t, v) for t, v in zip(timestamps, values) if start_time <= t <= end_time]
            if filtered_data:
                timestamps, values = zip(*filtered_data)
            else:
                timestamps, values = [], []
        
        # Convert timestamps to seconds for plotting
        timestamps_sec = [t / 1e9 for t in timestamps]
        
        # Plot each axis
        if values:
            values_array = np.array(values)
            if values_array.shape[1] >= 3:  # Ensure we have at least 3 dimensions
                axes[i].plot(timestamps_sec, values_array[:, 0], label='X', alpha=0.7)
                axes[i].plot(timestamps_sec, values_array[:, 1], label='Y', alpha=0.7)
                axes[i].plot(timestamps_sec, values_array[:, 2], label='Z', alpha=0.7)
            
        axes[i].set_title(f'{channel.replace("_", " ").title()}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel('Value')
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    plt.show()


def extract_imu_windows(mcap_file, window_size_ns=1e9, step_size_ns=None, channels=None):
    """
    Extract windowed IMU data for machine learning
    
    Args:
        mcap_file: Path to MCAP file
        window_size_ns: Window size in nanoseconds (default: 1 second)
        step_size_ns: Step size in nanoseconds (default: same as window_size_ns)
        channels: List of channel names to extract (default: raw sensor channels)
    
    Returns:
        List of dictionaries with 'timestamp', 'window_data', and 'labels'
    """
    if step_size_ns is None:
        step_size_ns = window_size_ns
    
    if channels is None:
        channels = ["mag_raw", "acc_raw", "gyro_raw"]
    
    data = read_synthetic_sensor_data(mcap_file, channels)
    
    # Find time range
    all_timestamps = []
    for channel_data in data.values():
        all_timestamps.extend([item[0] for item in channel_data])
    
    if not all_timestamps:
        return []
    
    start_time = min(all_timestamps)
    end_time = max(all_timestamps)
    
    windows = []
    current_time = start_time
    
    while current_time + window_size_ns <= end_time:
        window_start = current_time
        window_end = current_time + window_size_ns
        
        window_data = {}
        for channel in channels:
            # Extract data in this window
            channel_window = []
            for timestamp, values in data[channel]:
                if window_start <= timestamp < window_end:
                    channel_window.append((timestamp, values))
            window_data[channel] = channel_window
        
        windows.append({
            'timestamp': window_start,
            'window_data': window_data,
            'duration_ns': window_size_ns
        })
        
        current_time += step_size_ns
    
    return windows


def create_synthetic_dataset(
    file_pairs: List[List[str]],
    window_size_ns: int = 1e9,  # 1 second windows
    step_size_ns: int = 5e8,    # 0.5 second step (50% overlap)
    channels: List[str] = None,
):
    """
    Create a dataset from synthetic sensor data for classification
    
    Args:
        file_pairs: List of [mcap_file, labels_file] pairs
        window_size_ns: Window size in nanoseconds for sensor data segments
        step_size_ns: Step size in nanoseconds between windows
        channels: List of sensor channels to use (default: magnetometer raw)
    
    Returns:
        HuggingFace Dataset with input_values and labels
    """
    if channels is None:
        channels = ["mag_raw"]  # Focus on magnetometer for magnetic distortion detection
    
    all_events = {}
    for file_pair in file_pairs:
        mcap_file = file_pair[0]
        label_file = file_pair[1]
        with open(label_file, "rb") as f:
            json_data = json.load(f)
            all_events[mcap_file] = json_data["events"]
    
    # Extract unique labels from events
    labels = set()
    for events in all_events.values():
        for event in events:
            if event["metadata"] and "mag_distortion" in event["metadata"]:
                level = event["metadata"]["mag_distortion"]
                labels.add(f"mag_distortion_{level}")
    
    class_labels = ClassLabel(names=list(labels))
    
    # Define features - using a flattened sensor data array instead of spectrogram
    max_samples = int(window_size_ns / 1e6)  # Assume ~1kHz sampling rate
    features = Features(
        {
            "input_values": Array2D(
                dtype="float32", shape=(len(channels), max_samples)
            ),
            "labels": class_labels,
        }
    )
    
    input_data = []
    labels_list = []
    
    for mcap_file, events in all_events.items():
        print(f"Processing {mcap_file}...")
        
        # Read sensor data
        sensor_data = read_synthetic_sensor_data(mcap_file, channels)
        
        for event in events:
            metadata = event["metadata"]
            if metadata and "mag_distortion" in metadata:
                level = metadata["mag_distortion"]
                label = f"mag_distortion_{level}"
                
                start_time = event["startTime"]["sec"] * 1e9 + event["startTime"]["nsec"]
                end_time = event["endTime"]["sec"] * 1e9 + event["endTime"]["nsec"]
                
                print(f"Processing event: {label}, {start_time} to {end_time}")
                
                # Extract windowed data for this event
                current_time = start_time
                while current_time + window_size_ns <= end_time:
                    window_start = current_time
                    window_end = current_time + window_size_ns
                    
                    # Extract data for each channel in this window
                    window_data = np.zeros((len(channels), max_samples))
                    
                    for ch_idx, channel in enumerate(channels):
                        if channel in sensor_data:
                            channel_window = []
                            for timestamp, values in sensor_data[channel]:
                                if window_start <= timestamp < window_end:
                                    # Use magnitude of 3D sensor data
                                    if len(values) >= 3:
                                        magnitude = np.sqrt(values[0]**2 + values[1]**2 + values[2]**2)
                                        channel_window.append(magnitude)
                                    elif len(values) == 1:
                                        channel_window.append(values[0])
                            
                            # Pad or truncate to max_samples
                            if len(channel_window) > 0:
                                if len(channel_window) >= max_samples:
                                    window_data[ch_idx, :] = channel_window[:max_samples]
                                else:
                                    window_data[ch_idx, :len(channel_window)] = channel_window
                    
                    # Only add if we have some data
                    if np.any(window_data):
                        input_data.append(window_data)
                        labels_list.append(class_labels.str2int(label))
                    
                    current_time += step_size_ns
    
    print(f"Created {len(input_data)} training samples")
    return Dataset.from_dict(
        {
            "input_values": input_data,
            "labels": labels_list,
        },
        features=features,
    )


def create_dataset(
    file_pairs: List[List[str]],
    use_unlabeled_sections: bool = False,
    unlabeled_section_label: str = "Low",
    aggregate_labels: bool = False,
    aggregate_label_dict: dict = None,
):
    all_events = {}
    for file_pair in file_pairs:
        spectrogram_mcap_file = file_pair[0]
        label_file = file_pair[1]
        with open(label_file, "rb") as f:
            json_data = json.load(f)
            # add events to the existing list
            all_events[spectrogram_mcap_file] = json_data["events"]
        if use_unlabeled_sections:
            # open spectrogram mcap file header to get start and end time
            with open(spectrogram_mcap_file, "rb") as f:
                reader = make_reader(f)
                mcap_summary = reader.get_summary()
                mcap_start_time = mcap_summary.statistics.message_start_time
                mcap_end_time = mcap_summary.statistics.message_end_time

                current_file_events = sorted(
                    json_data["events"],
                    key=lambda x: x["startTime"]["sec"] * 1e9 + x["startTime"]["nsec"],
                )

                def create_gap_event(start_time, end_time, unlabeled_section_label):
                    return {
                        "endTime": {"sec": end_time // 1e9, "nsec": end_time % 1e9},
                        "startTime": {
                            "sec": start_time // 1e9,
                            "nsec": start_time % 1e9,
                        },
                        "metadata": {"voltage_level": unlabeled_section_label, "": ""},
                        "formattedTime": f"{start_time // 1e9}.{start_time % 1e9}",
                    }

                current_file_events = sorted(
                    json_data["events"],
                    key=lambda x: x["startTime"]["sec"] * 1e9 + x["startTime"]["nsec"],
                )
                gaps = []

                # Add gap at the beginning if necessary
                if current_file_events and (gap_start_time := mcap_start_time) < (
                    gap_end_time := current_file_events[0]["startTime"]["sec"] * 1e9
                    + current_file_events[0]["startTime"]["nsec"]
                ):
                    gaps.append(
                        create_gap_event(
                            gap_start_time, gap_end_time, unlabeled_section_label
                        )
                    )

                # Iterate over events to find gaps between events
                gaps.extend(
                    [
                        create_gap_event(
                            gap_start_time, gap_end_time, unlabeled_section_label
                        )
                        for i in range(1, len(current_file_events))
                        if (
                            gap_start_time := current_file_events[i - 1]["endTime"][
                                "sec"
                            ]
                            * 1e9
                            + current_file_events[i - 1]["endTime"]["nsec"]
                        )
                        < (
                            gap_end_time := current_file_events[i]["startTime"]["sec"]
                            * 1e9
                            + current_file_events[i]["startTime"]["nsec"]
                        )
                    ]
                )

                # Add gap at the end if necessary
                if current_file_events and (
                    gap_start_time := current_file_events[-1]["endTime"]["sec"] * 1e9
                    + current_file_events[-1]["endTime"]["nsec"]
                ) < (gap_end_time := mcap_end_time):
                    gaps.append(
                        create_gap_event(
                            gap_start_time, gap_end_time, unlabeled_section_label
                        )
                    )

                all_events[spectrogram_mcap_file].extend(gaps)

    # Create class labels from events
    labels = set()
    for events in all_events.values():
        for event in events:
            if event["metadata"]:
                key = list(event["metadata"].keys())[0]
                value = event["metadata"][key]
                if (
                    aggregate_labels
                    and aggregate_label_dict
                    and value in aggregate_label_dict
                ):
                    value = aggregate_label_dict[value]
                    #write the new value back to the metadata
                    event["metadata"] = {key: value}
                labels.add(f"{key}_{value}")

    class_labels = ClassLabel(names=list(labels))
    # Define features with audio and label columns
    features = Features(
        {
            "input_values": Array2D(
                dtype="float32", shape=(128, 1024)
            ),  # Define the spectrogram feature
            "labels": class_labels,  # Assign the class labels,
        }
    )
    spectrogram_data = []
    labels_list = []

    # iterate over all_events
    for spectrogram_mcap_file, events in all_events.items():
        with open(spectrogram_mcap_file, "rb") as mcap_f:
            reader = make_reader(mcap_f)
            mcap_summary = reader.get_summary()
            print_mcap_summary(mcap_summary)

            for event in events:
                metadata = event["metadata"]
                if len(metadata) != 0:
                    key = list(metadata.keys())[0]
                    value = metadata[key]
                    label = f"{key}_{value}"

                    start_time = (
                        event["startTime"]["sec"] * 1e9 + event["startTime"]["nsec"]
                    )
                    end_time = event["endTime"]["sec"] * 1e9 + event["endTime"]["nsec"]
                    print(label, start_time, end_time)

                    # Iterate over input_values messages in range start_time to end_time
                    for schema, channel, message in reader.iter_messages(
                        start_time=start_time, end_time=end_time
                    ):
                        json_str = message.data.decode("utf8").replace("'", '"')
                        json_data = json.loads(json_str)
                        if channel.topic == "input_values":
                            input_values_base64 = json_data["data"]

                            input_values = deserialize_numpy_array(input_values_base64)

                            spectrogram_data.append(input_values)
                            labels_list.append(class_labels.str2int(label))

    return Dataset.from_dict(
        {
            "input_values": spectrogram_data,
            "labels": labels_list,
        },
        features=features,
    )


def classify_from_spectrogram(spectrogram_mcap_file, classification_file, model):
    model.eval()
    with open(classification_file, "wb") as classification_f:
        writer = Writer(classification_f)

        # register the schema for the compressed image
        classification_schema_id = writer.register_schema(
            name="classification",
            encoding="jsonschema",
            data=json.dumps(
                {
                    "title": "Magnetic Distortion Classification",
                    "description": "output of the magnetic distortion classification model",
                    "type": "object",
                    "properties": {
                        "predicted_class_id": {
                            "type": "number",
                            "description": "predicted class id",
                        },
                        "label": {
                            "type": "string",
                            "description": "predicted class label",
                        },
                    },
                }
            ).encode(),
        )

        # register the channel for the compressed image
        classification_channel_id = writer.register_channel(
            "classification",
            "json",
            schema_id=classification_schema_id,
        )

        writer.start()

        with open(spectrogram_mcap_file, "rb") as mcap_f:
            reader = make_reader(mcap_f)
            mcap_summary = reader.get_summary()
            print_mcap_summary(mcap_summary)

            # Extract start and end times from the summary
            start_time = mcap_summary.statistics.message_start_time
            end_time = mcap_summary.statistics.message_end_time

            print(f"Start time: {start_time}")
            print(f"End time: {end_time}")
            for schema, channel, message in reader.iter_messages(
                start_time=start_time, end_time=end_time
            ):
                json_str = message.data.decode("utf8").replace("'", '"')
                json_data = json.loads(json_str)
                if channel.topic == "input_values":
                    input_values_base64 = json_data["data"]

                    input_values = deserialize_numpy_array(input_values_base64)
                    # Convert numpy array to PyTorch tensor
                    input_values_tensor = torch.from_numpy(input_values).unsqueeze(0)

                    # Perform classification on the spectrogram
                    with torch.no_grad():
                        outputs = model.forward(input_values=input_values_tensor)
                        logits = outputs.logits
                        predicted_class_id = logits.argmax().item()
                        predicted_class = model.config.id2label[predicted_class_id]
                        print(predicted_class, outputs.logits)

                    # Write the classification to the classification mcap
                    writer.add_message(
                        classification_channel_id,
                        log_time=message.log_time,
                        publish_time=message.publish_time,
                        data=json.dumps(
                            {
                                "predicted_class_id": predicted_class_id,
                                "label": predicted_class,
                            },
                        ).encode(),
                    )

        writer.finish()


def spectrogram_from_timeseries(
    input_file,
    spectrogram_mcap_file,
    sampling_rate=140,
    feature_extractor=None,
    window_size_ns=3e9,
    step_size_ns=10e9,
):
    with open(spectrogram_mcap_file, "wb") as spectrogram_mcap_f:
        writer = Writer(spectrogram_mcap_f)

        # register the schema for the compressed image
        spectrogram_schema_id = writer.register_schema(
            name="foxglove.CompressedImage",
            encoding="jsonschema",
            data=json.dumps(
                {
                    "title": "foxglove.CompressedImage",
                    "description": "A compressed image",
                    "$comment": "Generated by https://github.com/foxglove/schemas",
                    "type": "object",
                    "properties": {
                        "timestamp": {
                            "type": "object",
                            "title": "time",
                            "properties": {
                                "sec": {"type": "integer", "minimum": 0},
                                "nsec": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 999999999,
                                },
                            },
                            "description": "Timestamp of image",
                        },
                        "frame_id": {
                            "type": "string",
                            "description": "Frame of reference for the image. The origin of the frame is the optical center of the camera. +x points to the right in the image, +y points down, and +z points into the plane of the image.",
                        },
                        "data": {
                            "type": "string",
                            "contentEncoding": "base64",
                            "description": "Compressed image data",
                        },
                        "format": {
                            "type": "string",
                            "description": "Image format\n\nSupported values: image media types supported by Chrome, such as `webp`, `jpeg`, `png`",
                        },
                    },
                }
            ).encode(),
        )

        # register the channel for the compressed image
        spectrogram_channel_id = writer.register_channel(
            "spectrogram",
            "json",
            schema_id=spectrogram_schema_id,
        )

        input_values_schema_id = writer.register_schema(
            name="input_values",
            encoding="jsonschema",
            data=json.dumps(
                {
                    "title": "input_values",
                    "description": "Input array for classification",
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "contentEncoding": "base64",
                            "description": "base64 encoded input_values",
                        },
                    },
                }
            ).encode(),
        )

        input_values_channel_id = writer.register_channel(
            "input_values", "json", schema_id=input_values_schema_id
        )

        writer.start()

        with open(input_file, "rb") as mcap_f:
            reader = make_reader(mcap_f)
            mcap_summary = reader.get_summary()
            print_mcap_summary(mcap_summary)

            # Extract start and end times from the summary
            start_time = mcap_summary.statistics.message_start_time
            end_time = mcap_summary.statistics.message_end_time

            print(f"Start time: {start_time}")
            print(f"End time: {end_time}")

            # Iterate over segments within the range start_time to end_time
            current_time: int = start_time
            segment_index = 0
            while current_time + window_size_ns <= end_time:
                segment_start_time = current_time
                segment_end_time = current_time + window_size_ns
                print(
                    f"Processing segment {segment_index}: {segment_start_time} to {segment_end_time}"
                )
                segment_index += 1
                current_time += step_size_ns

                values = [[], [], []]
                for schema, channel, message in reader.iter_messages(
                    start_time=segment_start_time, end_time=segment_end_time
                ):
                    json_str = message.data.decode("utf8").replace("'", '"')
                    json_data = json.loads(json_str)
                    
                    # Use magnetometer raw data for spectrograms (3-axis magnetic field)
                    if channel.topic == "mag_raw":
                        values[0].append(json_data["values"][0])  # mag X
                        values[1].append(json_data["values"][1])  # mag Y
                        values[2].append(json_data["values"][2])  # mag Z

                waveform = torch.cat(
                    [
                        torch.tensor(values[0]),
                        torch.tensor(values[1]),
                        torch.tensor(values[2]),
                    ]
                )
                # print("waveform.shape", waveform.shape)

                # Resample the waveform to 16000 Hz
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate, new_freq=16000
                )
                waveform_resampled = resampler(waveform)

                spectrogram = feature_extractor(
                    waveform_resampled,
                    sampling_rate=feature_extractor.sampling_rate,
                    return_tensors="pt",
                )
                input_values = spectrogram.input_values

                # Assuming input_values is a PyTorch tensor
                input_values_np = (
                    input_values.squeeze().numpy()
                ).T  # Convert to NumPy array and remove batch dimension if present

                input_values_base64 = serialize_numpy_array(input_values_np)

                # Create the plot
                fig, ax = plt.subplots()
                ax.imshow(input_values_np, aspect="auto", origin="lower")

                # Remove borders and axes
                ax.axis("off")

                # Capture the plot output as a PNG image
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                buf.seek(0)

                # Encode the PNG image as a base64 string
                image_base64 = base64.b64encode(buf.read()).decode("utf-8")

                log_time = int(segment_end_time)
                publish_time = log_time
                timestamp_sec = int(log_time / 1e9)
                timestamp_nsec = int(log_time % 1e9)
                writer.add_message(
                    spectrogram_channel_id,
                    log_time=log_time,
                    data=json.dumps(
                        {
                            "data": image_base64,
                            "format": "png",
                            "id": "spectrogram",
                            "timestamp": {
                                "nsec": timestamp_nsec,
                                "sec": timestamp_sec,
                            },
                        }
                    ).encode(),
                    publish_time=publish_time,
                )
                writer.add_message(
                    input_values_channel_id,
                    log_time=log_time,
                    data=json.dumps({"data": input_values_base64}).encode(),
                    publish_time=publish_time,
                )

        # finish writing the spectrogram mcap
        writer.finish()
