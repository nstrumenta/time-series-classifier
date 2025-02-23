import json
from mcap.reader import make_reader
from mcap.writer import Writer
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import numpy as np
import os
import datetime
import torch, torchaudio
from typing import Dict, List, Optional
from datasets import Dataset, Value, Array2D, ClassLabel, Features
from transformers import ASTFeatureExtractor
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


def create_dataset(
    file_pairs: List[List[str]],
):
    all_events = {}
    for file_pair in file_pairs:
        spectrogram_mcap_file = file_pair[0]
        label_file = file_pair[1]
        with open(label_file, "rb") as f:
            json_data = json.load(f)
            # add events to the existing list
            all_events[spectrogram_mcap_file] = json_data["events"]

    # Create class labels from events
    labels = {
        f"{list(event['metadata'].keys())[0]}_{event['metadata'][list(event['metadata'].keys())[0]]}"
        for events in all_events.values()
        for event in events
        if event["metadata"]
    }
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
                    "title": "Peek Classification",
                    "description": "output of the classification model",
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
                    if channel.topic == "PEEK_RAW":
                        values[0].append(json_data["values"][3] / 4000)
                        values[1].append(json_data["values"][4] / 4000)
                        values[2].append(json_data["values"][5] / 4000)

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
