"""
Common utilities for scripts in the time-series-classifier project.
Provides shared setup functions for NstrumentaClient, path management, 
and file operations.
"""

import os
import sys
import tarfile
import hashlib
from typing import Optional
from nstrumenta import NstrumentaClient

# Capture the directory where the script was started
_initial_cwd = os.getcwd()


def setup_src_path() -> str:
    """
    Add the src directory to Python path for imports.
    
    Returns:
        str: Absolute path to the src directory
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, "..", "src"))
    
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    
    return src_dir


def setup_nstrumenta_client() -> NstrumentaClient:
    """
    Initialize and return a NstrumentaClient with API key from environment.
    Handles both Google Colab and regular environment setups.
    
    Returns:
        NstrumentaClient: Configured client instance
    """
    # Handle Google Colab environment
    if "google.colab" in sys.modules:
        from google.colab import userdata
        os.environ["NSTRUMENTA_API_KEY"] = userdata.get("NSTRUMENTA_API_KEY")
    
    client = NstrumentaClient(os.getenv("NSTRUMENTA_API_KEY"))
    
    # Print project info for verification
    try:
        print(f"Connected to project: {client.get_project()}")
    except Exception as e:
        print(f"Warning: Could not get project info: {e}")
    
    return client


def reset_to_initial_cwd():
    """Reset current working directory to where the script was initially run."""
    os.chdir(_initial_cwd)
    print(f"Current working directory reset to: {os.getcwd()}")


def setup_working_directory(path: str) -> str:
    """
    Create working directory if it doesn't exist and change to it.
    
    Args:
        path: Path to the working directory
        
    Returns:
        str: Absolute path to the working directory
    """
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    os.chdir(abs_path)
    print(f"Working directory set to: {os.getcwd()}")
    return abs_path


def fetch_nstrumenta_file(
    client: NstrumentaClient, 
    remote_file: str, 
    local_path: Optional[str] = None, 
    extract_tar: bool = False
) -> str:
    """
    Download a file from Nstrumenta if it doesn't already exist locally.
    
    Args:
        client: NstrumentaClient instance
        remote_file: Remote file name/path to download
        local_path: Local path to save to (defaults to remote_file name)
        extract_tar: Whether to extract if it's a tar.gz file
        
    Returns:
        str: Path to the local file
    """
    local_path = local_path or remote_file
    
    if not os.path.exists(local_path):
        print(f"Downloading {remote_file} to {local_path}")
        client.download(remote_file, local_path)
        
        if extract_tar and local_path.endswith(('.tar.gz', '.tgz')):
            print(f"Extracting {local_path}")
            with tarfile.open(local_path, "r:gz") as tar:
                tar.extractall()
    else:
        print(f"{local_path} already exists")
    
    return local_path


import hashlib


def upload_with_prefix(
    client: NstrumentaClient, 
    local_file: str, 
    remote_prefix: str,
    remote_filename: str = None,
    overwrite: bool = True
):
    """
    Upload a file with a remote path prefix.
    
    Args:
        client: NstrumentaClient instance
        local_file: Local file path to upload
        remote_prefix: Remote path prefix (e.g., log name)
        remote_filename: Optional remote filename (defaults to basename of local_file)
        overwrite: Whether to overwrite existing remote files
    """
    if remote_filename is None:
        remote_filename = os.path.basename(local_file)
    
    remote_path = f"{remote_prefix}/{remote_filename}"
    print(f"Uploading {local_file} to {remote_path}")
    client.upload(local_file, remote_path, overwrite=overwrite)


def file_hash(filepath: str) -> str:
    """
    Calculate MD5 hash of a file.
    
    Args:
        filepath: Path to the file
        
    Returns:
        MD5 hash as hex string
    """
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except FileNotFoundError:
        return ""


def upload_if_changed(
    client: NstrumentaClient,
    local_file: str,
    remote_prefix: str,
    hash_cache: dict = None,
    overwrite: bool = True
) -> bool:
    """
    Upload a file only if it has changed since last upload.
    
    Args:
        client: NstrumentaClient instance
        local_file: Local file path to upload
        remote_prefix: Remote path prefix (e.g., log name)
        hash_cache: Dictionary to store file hashes (optional)
        overwrite: Whether to overwrite existing remote files
        
    Returns:
        True if file was uploaded, False if skipped
    """
    if not os.path.exists(local_file):
        return False
        
    current_hash = file_hash(local_file)
    cache_key = f"{remote_prefix}/{local_file}"
    
    # Check if we have a hash cache and if the file hasn't changed
    if hash_cache is not None and cache_key in hash_cache:
        if hash_cache[cache_key] == current_hash:
            print(f"Skipping {local_file} - unchanged")
            return False
    
    # File is new or changed, upload it
    upload_with_prefix(client, local_file, remote_prefix, overwrite=overwrite)
    
    # Update hash cache
    if hash_cache is not None:
        hash_cache[cache_key] = current_hash
        
    return True


# Convenience function for common script initialization
def init_script_environment() -> tuple[str, NstrumentaClient]:
    """
    Perform common script initialization: setup src path and Nstrumenta client.
    
    Returns:
        tuple: (src_dir_path, nstrumenta_client)
    """
    src_dir = setup_src_path()
    client = setup_nstrumenta_client()
    return src_dir, client
