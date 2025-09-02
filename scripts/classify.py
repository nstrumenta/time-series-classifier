import os
from script_utils import (
    init_script_environment,
    setup_working_directory,
    reset_to_initial_cwd,
    fetch_nstrumenta_file,
    upload_with_prefix
)

# Initialize script environment
src_dir, nst_client = init_script_environment()

# Import project modules after src path is set up
import mcap_utilities


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
reset_to_initial_cwd()
# change to the working folder
setup_working_directory(working_folder)


fetch_nstrumenta_file(nst_client, input_file)
fetch_nstrumenta_file(nst_client, model_tar_filename, model_tar_filename, extract_tar=True)


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
        upload_with_prefix(
            nst_client,
            spectrogram_mcap_file,
            log_prefix,
            overwrite=True
        )
    else:
        print(f"{spectrogram_mcap_file} exists.")


create_spectrogram_if_not_exists(input_file, spectrogram_mcap_file)

mcap_utilities.classify_from_spectrogram(
    spectrogram_mcap_file=spectrogram_mcap_file,
    classification_file=classification_file,
    model=model,
)

upload_with_prefix(
    nst_client,
    classification_file,
    log_prefix,
    overwrite=True
)
