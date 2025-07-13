import sys
import os
from nstrumenta import NstrumentaClient
import tarfile


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


# Function to reset the cwd to the initial directory
def reset_cwd():
    os.chdir(initial_cwd)
    print(f"Current working directory reset to: {os.getcwd()}")


model_id = "3AF306"

# log_prefix = "Sensor_Log_2023-11-08_07_36_21"
# log_prefix = "Sensor_Log_2023-11-08_08_25_49"
# log_prefix = "Sensor_Log_2023-12-07_10_20_28"
log_prefix = "Sensor_Log_2023-11-08_09_05_43"

working_folder = f"./temp/{model_id}"
input_file = f"{log_prefix}.mcap"
spectrogram_mcap_file = f"{log_prefix}.spectrogram.mcap"
classification_file = f"{log_prefix}.classification.mcap"
model_tar_filename = f"{model_id}.model.tar.gz"

# set working folder to the project root
reset_cwd()
# change to the working folder
os.makedirs(working_folder, exist_ok=True)
os.chdir(working_folder)


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


download_if_not_exists(input_file)
download_if_not_exists(model_tar_filename, model_tar_filename, extract=True)


from transformers import ASTFeatureExtractor
from transformers import ASTForAudioClassification

# we define which pretrained model we want to use and instantiate a feature extractor
pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)

# load the fine_tuned model from "model" for prediction using huggingface transformers
model = ASTForAudioClassification.from_pretrained("./model")


def create_spectrogram_if_not_exists(input_file, spectrogram_mcap_file):
    if not os.path.exists(spectrogram_mcap_file):
        mcap_utilities.spectrogram_from_timeseries(
            input_file=input_file,
            spectrogram_mcap_file=spectrogram_mcap_file,
            feature_extractor=feature_extractor,
        )
        nst_client.upload(
            spectrogram_mcap_file,
            f"{log_prefix}/{spectrogram_mcap_file}",
            overwrite=True,
        )
    else:
        print(f"{spectrogram_mcap_file} exists.")


create_spectrogram_if_not_exists(input_file, spectrogram_mcap_file)

mcap_utilities.classify_from_spectrogram(
    spectrogram_mcap_file=spectrogram_mcap_file,
    classification_file=classification_file,
    model=model,
)

nst_client.upload(
    classification_file,
    f"{log_prefix}/{classification_file}",
    overwrite=True,
)
