"""
Dataset creation utilities for MCAP data
"""

import json
import io
import base64
import numpy as np
from mcap.reader import make_reader
from datasets import Dataset, Array2D, ClassLabel, Features
from typing import List, Dict, Optional
from .reader import print_mcap_summary


def serialize_numpy_array(array: np.ndarray) -> str:
    """Serialize a numpy array to a base64 string"""
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=True)
    buffer.seek(0)
    array_bytes = buffer.read()
    array_base64 = base64.b64encode(array_bytes).decode("utf-8")
    return array_base64


def deserialize_numpy_array(array_base64: str) -> np.ndarray:
    """Deserialize a base64 string back to a numpy array"""
    array_bytes = base64.b64decode(array_base64)
    buffer = io.BytesIO(array_bytes)
    array = np.load(buffer, allow_pickle=True)
    return array


def create_dataset(
    file_pairs: List[List[str]],
    use_unlabeled_sections: bool = False,
    unlabeled_section_label: str = "Low",
    aggregate_labels: bool = False,
    aggregate_label_dict: Optional[Dict[str, str]] = None,
) -> Dataset:
    """
    Create a dataset from MCAP files and label files
    
    Args:
        file_pairs: List of [mcap_file, label_file] pairs
        use_unlabeled_sections: Whether to include unlabeled sections
        unlabeled_section_label: Label for unlabeled sections
        aggregate_labels: Whether to aggregate labels using mapping
        aggregate_label_dict: Dictionary for label aggregation
    
    Returns:
        HuggingFace Dataset object
    """
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
