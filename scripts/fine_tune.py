import sys
import os
import numpy as np
from datasets import Audio, ClassLabel, load_dataset
import torch
from nstrumenta import NstrumentaClient
import tarfile
import uuid


# Determine the absolute path to the src directory
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, "..", "src"))
# Add the src directory to the Python path
sys.path.append(src_dir)

import mcap_utilities


# use colab user data or getenv
if "google.colab" in sys.modules:
    from google.colab import userdata

    os.environ["NSTRUMENTA_API_KEY"] = userdata.get("NSTRUMENTA_API_KEY")

nst_client = NstrumentaClient(os.getenv("NSTRUMENTA_API_KEY"))

print(nst_client.get_project())

# Store the initial working directory
initial_cwd = os.getcwd()


# create a model_id to name this fine_tuned model
model_id = "3AF306"  # uuid.uuid4().hex.upper()[:6]

from transformers import ASTFeatureExtractor

# we define which pretrained model we want to use and instantiate a feature extractor
pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)

working_folder = f"./temp/{model_id}"


# set working folder to the project root
# Function to reset the cwd to the initial directory
def reset_cwd():
    os.chdir(initial_cwd)
    print(f"Current working directory reset to: {os.getcwd()}")


reset_cwd()

# change to the working folder
os.makedirs(working_folder, exist_ok=True)
os.chdir(working_folder)
file_pairs = []

# print the current working directory
print(f"current working directory: {os.getcwd()}")


def download_if_not_exists(file, dest=None, extract=False):
    dest = dest if dest else file
    if not os.path.exists(dest):
        print(f"downloading {file} to {dest}.")
        nst_client.download(file, dest)
        if extract:
            with tarfile.open(dest, "r:gz") as tar:
                tar.extractall()

    else:
        print(f"{dest} exists.")


logs = [
    "Sensor_Log_2023-11-08_07_36_21",
    "Sensor_Log_2023-11-08_08_25_49",
    "Sensor_Log_2023-12-07_10_20_28",
    "Sensor_Log_2023-11-08_09_05_43",
]
for log_prefix in logs:
    # download the input file and label file
    input_file = f"{log_prefix}.mcap"
    label_file = f"{log_prefix}.labels.json"
    spectrogram_mcap_file = f"{log_prefix}.spectrogram.mcap"
    file_pairs.append([spectrogram_mcap_file, label_file])
    download_if_not_exists(input_file)
    download_if_not_exists(label_file)

    def create_spectrogram_if_not_exists(input_file, spectrogram_mcap_file):
        if not os.path.exists(spectrogram_mcap_file):
            mcap_utilities.spectrogram_from_timeseries(
                input_file=input_file,
                spectrogram_mcap_file=spectrogram_mcap_file,
                feature_extractor=feature_extractor,
            )
            nst_client.upload(
                spectrogram_mcap_file,
                f"{model_id}/{spectrogram_mcap_file}",
                overwrite=True,
            )
        else:
            print(f"{spectrogram_mcap_file} exists.")

    create_spectrogram_if_not_exists(input_file, spectrogram_mcap_file)


# create a dataset from the file pairs [spectrogram_file, label_file]
dataset = mcap_utilities.create_dataset(
    file_pairs=file_pairs,
    use_unlabeled_sections=True,
    unlabeled_section_label="low",
    aggregate_labels=True,
    aggregate_label_dict={
        "10_kV": "medium",
        "20_kV": "medium",
        "110_kV": "high",
        "130_kV": "high",
    },
)

dataset.save_to_disk("dataset")

from transformers import ASTConfig, ASTForAudioClassification

# Load configuration from the pretrained model
config = ASTConfig.from_pretrained(pretrained_model)

# Access the ClassLabel feature for the labels
label_feature = dataset.features["labels"]

# Get the label names
label_names = label_feature.names

print("Label names:", label_names)

config.num_labels = len(label_names)
config.label2id = {label: i for i, label in enumerate(label_names)}
config.id2label = {i: label for label, i in config.label2id.items()}


# split training data
if "test" not in dataset:
    dataset = dataset.train_test_split(
        test_size=0.2, shuffle=True, seed=0, stratify_by_column="labels"
    )

# Initialize the model with the updated configuration
model = ASTForAudioClassification.from_pretrained(
    pretrained_model, config=config, ignore_mismatched_sizes=True
)
model.init_weights()

from transformers import TrainingArguments

# Configure training run with TrainingArguments class
training_args = TrainingArguments(
    output_dir="runs",
    logging_dir="logs",
    report_to="tensorboard",
    learning_rate=5e-5,  # Learning rate
    push_to_hub=False,
    num_train_epochs=5,  # Number of epochs
    per_device_train_batch_size=8,  # Batch size per device
    eval_strategy="epoch",  # Evaluation strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_strategy="steps",
    logging_steps=20,
)

import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")
recall = evaluate.load("recall")
precision = evaluate.load("precision")
f1 = evaluate.load("f1")

AVERAGE = "macro" if config.num_labels > 2 else "binary"


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    predictions = np.argmax(logits, axis=1)
    metrics = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
    metrics.update(
        precision.compute(
            predictions=predictions, references=eval_pred.label_ids, average=AVERAGE
        )
    )
    metrics.update(
        recall.compute(
            predictions=predictions, references=eval_pred.label_ids, average=AVERAGE
        )
    )
    metrics.update(
        f1.compute(
            predictions=predictions, references=eval_pred.label_ids, average=AVERAGE
        )
    )
    return metrics


from transformers import Trainer

# Setup the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,  # Use the metrics function from above
)

# print command to start tensorboard in another terminal
print(f"tensorboard --logdir={training_args.logging_dir}")

trainer.train()

# save trained model
trainer.save_model("model")

# package model folder up as tarball
model_tar_filename = f"{model_id}.model.tar.gz"
print(f"packaging model folder into {model_tar_filename}")
with tarfile.open(model_tar_filename, "w:gz") as tar:
    tar.add("model", arcname=os.path.basename("model"))

# upload model to nstrumenta
print(f"uploading {model_tar_filename} to nstrumenta.")
nst_client.upload(model_tar_filename, f"{model_tar_filename}", overwrite=True)

# run inference on test set
predictions = trainer.predict(dataset["test"])
print(predictions)

# print some training metrics
metrics = trainer.evaluate()
print(metrics)
