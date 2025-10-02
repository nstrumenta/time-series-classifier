import sys
import os
import argparse
import numpy as np
from datasets import Audio, ClassLabel, load_dataset
import torch
import uuid
import tarfile
from transformers import ASTFeatureExtractor
from script_utils import (
    init_script_environment,
    setup_working_directory,
    reset_to_initial_cwd,
    fetch_nstrumenta_file,
    upload_with_prefix,
    upload_if_changed
)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Fine-tune magnetic distortion classifier')
parser.add_argument('--debug', action='store_true', 
                    help='Run in debug mode with reduced training parameters')
args = parser.parse_args()

# Initialize script environment
src_dir, nst_client = init_script_environment()

# Import project modules after src path is set up
import mcap_utilities
from synthetic import SyntheticDataGenerator

# Configuration based on debug mode
if args.debug:
    model_id = "MAG_DIST_DEBUG"
    max_steps = 10
    num_train_epochs = 1
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1
    eval_steps = 5
    debug_limit = 1  # Only use first sequence for debugging
    print("=== DEBUG MODE ENABLED ===")
    print("Using reduced parameters for fast testing")
else:
    model_id = "MAG_DIST_01"
    max_steps = 300
    num_train_epochs = 3
    per_device_train_batch_size = 2
    per_device_eval_batch_size = 4
    eval_steps = 50
    debug_limit = None  # Use all sequences
    print("=== FULL TRAINING MODE ===")
    print("Using full parameters for production training")

from transformers import ASTFeatureExtractor

# we define which pretrained model we want to use and instantiate a feature extractor
pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)

working_folder = f"./temp/{model_id}"

reset_to_initial_cwd()

# change to the working folder
setup_working_directory(working_folder)

print("=== Using Pre-generated Synthetic Training Data ===")

# Use all training sequence files from the synthetic_datasets directory
synthetic_data_dir = "../synthetic_datasets"
mcap_files = [f for f in os.listdir(synthetic_data_dir) 
              if f.startswith("training_sequence_") and f.endswith(".mcap") and not f.endswith(".spectrogram.mcap")]

# Apply debug limit if specified
if debug_limit is not None:
    mcap_files = mcap_files[:debug_limit]
    print(f"Found {len(mcap_files)} sequence files for debugging")
else:
    print(f"Found {len(mcap_files)} sequence files")

file_pairs = []
upload_hash_cache = {}  # Cache for tracking file changes

for mcap_file in mcap_files:
    mcap_path = os.path.join(synthetic_data_dir, mcap_file)
    labels_file = mcap_path.replace(".mcap", ".labels.json")
    
    # Extract plan name from file (e.g., training_sequence_2.mcap -> training_sequence_2)
    plan_name = mcap_file.replace(".mcap", "")
    remote_prefix = f"synthetic_datasets/{plan_name}"  # Upload to individual subdirectories
    
    # Generate spectrograms in the same directory as the source data
    spectrogram_mcap_file = os.path.join(synthetic_data_dir, f"{plan_name}.spectrogram.mcap")
    print(f"Processing {mcap_file}...")
    mcap_utilities.spectrogram_from_timeseries(mcap_path, spectrogram_mcap_file, feature_extractor=feature_extractor)
    print(f"✓ Created spectrogram: {spectrogram_mcap_file}")
    
    # Upload spectrogram if it has changed
    spectrogram_filename = os.path.basename(spectrogram_mcap_file)
    
    # Use a temporary copy in current directory for upload
    temp_spectrogram = spectrogram_filename
    import shutil
    shutil.copy2(spectrogram_mcap_file, temp_spectrogram)
    
    uploaded = upload_if_changed(nst_client, temp_spectrogram, remote_prefix, upload_hash_cache)
    if uploaded:
        print(f"📤 Uploaded to: {remote_prefix}/{spectrogram_filename}")
    
    # Clean up temp file
    if os.path.exists(temp_spectrogram):
        os.remove(temp_spectrogram)
    
    file_pairs.append([spectrogram_mcap_file, labels_file])

mode_name = "DEBUG" if args.debug else "FULL TRAINING"
print(f"\n=== Ready to train with {len(file_pairs)} synthetic datasets ({mode_name}) ===")

# create a dataset from the synthetic file pairs [spectrogram_file, label_file]
print("\n=== Creating Training Dataset ===")
dataset = mcap_utilities.create_dataset(
    file_pairs=file_pairs,
    use_unlabeled_sections=False,  # Use only labeled sections for magnetic distortion
    aggregate_labels=False,  # Keep original mag_distortion labels (0, 1, 2)
)

dataset.save_to_disk("dataset")

from transformers import ASTConfig, ASTForAudioClassification

# Load configuration from the pretrained model
config = ASTConfig.from_pretrained(pretrained_model)

# Get unique labels from the dataset (labels are now stored as integers)
# Note: After Array2D fix, we use direct label mapping instead of ClassLabel
unique_labels = sorted(set(dataset["labels"]))
num_labels = len(unique_labels)

print(f"Magnetic distortion classification labels: {unique_labels}")
print("Label mapping: 0=none, 1=low, 2=high")

config.num_labels = num_labels
config.label2id = {str(i): i for i in unique_labels}
config.id2label = {i: str(i) for i in unique_labels}

# split training data
if "test" not in dataset:
    # Note: stratify_by_column requires ClassLabel, which we removed for Array2D fix
    # We'll do manual stratified split instead
    from sklearn.model_selection import train_test_split
    
    # Get indices for stratified split
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=0.2, 
        shuffle=True, 
        random_state=0,
        stratify=dataset["labels"]
    )
    
    # Create train/test splits using indices
    dataset = {
        "train": dataset.select(train_idx),
        "test": dataset.select(test_idx)
    }

# Initialize the model with the updated configuration for magnetic distortion classification
model = ASTForAudioClassification.from_pretrained(
    pretrained_model, config=config, ignore_mismatched_sizes=True
)
model.init_weights()

from transformers import TrainingArguments

# Configure training run with TrainingArguments class
training_args = TrainingArguments(
    output_dir=f"runs{'_debug' if args.debug else ''}",
    logging_dir=f"logs{'_debug' if args.debug else ''}",
    report_to="tensorboard", 
    learning_rate=5e-5,  # Learning rate
    push_to_hub=False,
    max_steps=max_steps,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=eval_steps,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_strategy="steps",
    logging_steps=10,  # More frequent logging
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

# Custom data collator to reshape flattened spectrograms back to 2D
# Note: After Array2D fix, spectrograms are stored as flattened 1D arrays
def collate_fn(batch):
    """Reshape flattened spectrograms back to 2D (128, 1024)"""
    import torch
    
    input_values = []
    labels = []
    
    for item in batch:
        # Get flattened input and reshape to 2D
        flattened = item["input_values"]
        if isinstance(flattened, list):
            flattened = np.array(flattened)
        
        # Reshape from 1D (131072,) back to 2D (128, 1024)
        reshaped = flattened.reshape(128, 1024)
        input_values.append(reshaped)
        labels.append(item["labels"])
    
    return {
        "input_values": torch.tensor(np.array(input_values), dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

# Setup the trainer for magnetic distortion classification
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,  # Use the metrics function from above
    data_collator=collate_fn,  # Use custom collator to reshape spectrograms
)

# print command to start tensorboard in another terminal
mode_suffix = "_debug" if args.debug else ""
print(f"\n=== Training Magnetic Distortion Classifier ({mode_name}) ===")
print(f"Monitor training progress: tensorboard --logdir={training_args.logging_dir}")
if args.debug:
    print(f"This is a DEBUG run with max {max_steps} steps and {num_train_epochs} epoch")

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
nst_client.upload(model_tar_filename, model_tar_filename, overwrite=True)

# run inference on test set
print("\n=== Evaluating Model ===")
predictions = trainer.predict(dataset["test"])

# print training metrics
metrics = trainer.evaluate()
print(f"\n=== {mode_name} Complete ===")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

if args.debug:
    print(f"Check {training_args.logging_dir}/ for tensorboard logs")
    print(f"Check {training_args.output_dir}/ for model checkpoints")
else:
    print(f"Model saved and uploaded: {model_tar_filename}")
    print(f"Check {training_args.logging_dir}/ for tensorboard logs")

print(f"\n✅ Successfully trained magnetic distortion classifier: {model_id}")
print("Model can classify magnetic distortion levels: none (0), low (1), high (2)")
