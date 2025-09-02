"""
Spectrogram processing utilities for MCAP data
"""

import json
import io
import base64
import matplotlib.pyplot as plt
import torch
import torchaudio
from mcap.reader import make_reader
from mcap.writer import Writer
from typing import Optional
from .reader import print_mcap_summary
from .dataset import serialize_numpy_array, deserialize_numpy_array


def classify_from_spectrogram(spectrogram_mcap_file: str, classification_file: str, model):
    """
    Classify spectrograms using a trained model
    
    Args:
        spectrogram_mcap_file: Input MCAP file with spectrograms
        classification_file: Output MCAP file for classifications
        model: Trained classification model
    """
    model.eval()
    with open(classification_file, "wb") as classification_f:
        writer = Writer(classification_f)

        # register the schema for the classification
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

        # register the channel for the classification
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
    input_file: str,
    spectrogram_mcap_file: str,
    sampling_rate: int = 140,
    feature_extractor=None,
    window_size_ns: float = 3e9,
    step_size_ns: float = 10e9,
):
    """
    Convert time series data to spectrograms
    
    Args:
        input_file: Input MCAP file with time series data
        spectrogram_mcap_file: Output MCAP file with spectrograms
        sampling_rate: Sampling rate of input data
        feature_extractor: Feature extractor for spectrograms
        window_size_ns: Window size in nanoseconds
        step_size_ns: Step size in nanoseconds
    """
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
