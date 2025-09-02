"""
Example script showing how to use the common script utilities.
This demonstrates the standard pattern for scripts in this project.
"""

from script_utils import (
    init_script_environment,
    setup_working_directory, 
    reset_to_initial_cwd,
    fetch_nstrumenta_file,
    upload_with_prefix
)

# Initialize script environment (sets up src path and NstrumentaClient)
src_dir, nst_client = init_script_environment()

# Now you can import project modules
import mcap_utilities

def main():
    # Define your working parameters
    model_id = "EXAMPLE_MODEL"
    log_prefix = "example_log_2024_01_01"
    
    # Set up working directory
    working_folder = f"./temp/{model_id}"
    
    # Reset to initial directory, then set up working folder
    reset_to_initial_cwd()
    setup_working_directory(working_folder)
    
    # Download required files
    input_file = f"{log_prefix}.mcap"
    fetch_nstrumenta_file(nst_client, input_file)
    
    # Download and extract a model
    model_tar = f"{model_id}.model.tar.gz"
    fetch_nstrumenta_file(nst_client, model_tar, extract_tar=True)
    
    # Do your processing here...
    output_file = f"{log_prefix}.processed.mcap"
    # ... create output_file ...
    
    # Upload results
    upload_with_prefix(nst_client, output_file, log_prefix, overwrite=True)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
