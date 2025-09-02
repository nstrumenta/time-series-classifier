# Script Utilities

Common utilities for scripts in the time-series-classifier project.

## Usage

Import the required functions from `script_utils`:

```python
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
```

## Functions

### `init_script_environment() -> tuple[str, NstrumentaClient]`
- Sets up the src directory in Python path
- Initializes NstrumentaClient with API key from environment
- Handles both Google Colab and regular environments
- Returns: (src_directory_path, nstrumenta_client)

### `setup_working_directory(path: str) -> str`
- Creates the directory if it doesn't exist
- Changes current working directory to the specified path
- Returns the absolute path to the directory

### `reset_to_initial_cwd()`
- Resets current working directory to where the script was initially run

### `fetch_nstrumenta_file(client, remote_file, local_path=None, extract_tar=False) -> str`
- Downloads a file from Nstrumenta if it doesn't exist locally
- Optionally extracts tar.gz files
- Returns the local file path

### `upload_with_prefix(client, local_file, remote_prefix, overwrite=True)`
- Uploads a file with a remote path prefix
- Useful for organizing files by log name or model ID

## Common Script Pattern

```python
# 1. Initialize environment
src_dir, nst_client = init_script_environment()
import mcap_utilities

# 2. Define parameters
model_id = "MY_MODEL"
log_prefix = "my_log_2024_01_01"

# 3. Set up working directory
working_folder = f"./temp/{model_id}"
reset_to_initial_cwd()
setup_working_directory(working_folder)

# 4. Download required files
fetch_nstrumenta_file(nst_client, "input.mcap")
fetch_nstrumenta_file(nst_client, "model.tar.gz", extract_tar=True)

# 5. Process data
# ... your processing code ...

# 6. Upload results
upload_with_prefix(nst_client, "output.mcap", log_prefix)
```

## Migration

To migrate existing scripts:

1. Replace the common imports and setup code with `init_script_environment()`
2. Replace custom `download_if_not_exists` functions with `fetch_nstrumenta_file()`
3. Replace `nst_client.upload()` calls with `upload_with_prefix()` where appropriate
4. Replace manual directory setup with `setup_working_directory()`
5. Replace manual `reset_cwd()` with `reset_to_initial_cwd()`
