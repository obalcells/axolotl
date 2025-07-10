import torch
import os
import shutil
import tempfile
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, hf_hub_download
import sys
from pathlib import Path
import os
import argparse


PROBES_FOLDER = Path.home() / "probes"

def download_adapter_from_hf(
    base_model: PreTrainedModel,
    repo_id: str, 
    probe_id: str = "final_probe_20250515_193252",
    local_folder: str = None, 
    token: str = None,
    device: str = "auto",
    layer_idx: int = None
):
    """
    Downloads a LoRA model and probe head from Hugging Face Hub and loads them.
    
    Args:
        base_model: The base model to apply the LoRA adapter to
        repo_id: Repository ID in the format 'username/repo_name'
        repo_subfolder: Subfolder within the repository containing the model
        local_folder: Path to a local folder for downloading (if None, creates a temp folder)
        token: HF API token (optional)
        device: Device to load the models to ("cpu", "cuda", "auto", etc.)
        layer_idx: Optional layer index to use for the probe (if None, uses the layer from config)
        
    Returns:
        Tuple of (loaded_lora_model, loaded_probe)
    """
    # Validate repository ID
    # validate_repo_id(repo_id)

    repo_subfolder = f"value_head_probes/{probe_id}"
    local_folder = PROBES_FOLDER / probe_id if local_folder is None else local_folder

    # Create a temporary folder if local_folder is not provided
    temp_dir = None
    if local_folder is None:
        temp_dir = tempfile.mkdtemp()
        local_folder = temp_dir
        print(f"Created temporary folder: {local_folder}")
    
    # Download all the files in the given subfolder from the HuggingFace model repository
    try:
        # Initialize API
        api = HfApi()
        
        # Create local folder if it doesn't exist
        os.makedirs(local_folder, exist_ok=True)
        
        # List all files in the repository
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision="main")

        # Filter files by subfolder if specified
        if repo_subfolder:
            subfolder_files = [file for file in repo_files if file.startswith(repo_subfolder)]
        else:
            subfolder_files = repo_files

        print(subfolder_files)
        
        # Download each file
        for file_path in subfolder_files:
            # Get the relative path within the subfolder
            if repo_subfolder:
                relative_path = file_path[len(repo_subfolder):].lstrip('/')
            else:
                relative_path = file_path
            
            # Create subdirectory if needed
            local_subdir = os.path.join(local_folder, os.path.dirname(relative_path))
            os.makedirs(local_subdir, exist_ok=True)
            
            # Download the file to the correct local path
            local_file_path = os.path.join(local_folder, relative_path)

            # Use hf_hub_download to get the file
            downloaded_file = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                token=token
            )
            
            # Copy the downloaded file to the desired location
            print(f"Downloading {downloaded_file} to {local_file_path}")
            shutil.copy(downloaded_file, local_file_path)
        
        print(f"Files downloaded to {local_folder}: {os.listdir(local_folder)}")
        if len(os.listdir(local_folder)) == 0:
            raise ValueError(f"No files downloaded from {repo_id}/{repo_subfolder}")
        
        # Load LoRA model
        lora_model = PeftModel.from_pretrained(
            base_model,
            local_folder,
            device_map=device
        )

        return lora_model
    
    finally:
        # Clean up temporary directory if we created one
        if temp_dir and temp_dir != local_folder:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Cleaned up temporary folder: {temp_dir}")

parser = argparse.ArgumentParser()
parser.add_argument("--probe_id", type=str, default="llama3_1_8b")
args = parser.parse_args()

probe_id = args.probe_id

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="auto"
)

lora_model = download_adapter_from_hf(
    base_model, 
    repo_id="obalcells/hallucination-probes",
    probe_id=probe_id,
)

# lora_model = PeftModel.from_pretrained(
#     base_model,
#     PROBES_FOLDER / probe_id,
#     device_map="auto"
# )

print(lora_model)