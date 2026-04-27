#!/usr/bin/env python3
"""Upload file to ModelScope dataset."""

from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, Visibility
import os

# Configuration
file_path = "/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/nemotron_40000/nemotron_think_on_samples_40000_qwen3_8b_regen.jsonl"
repo_id = "alexwangsjtu/Nemotron_400000_Qwen3_think"
target_path = "nemotron_think_on_samples_40000_qwen3_8b_regen.jsonl"

# Initialize API
api = HubApi()

YOUR_ACCESS_TOKEN = 'ms-45fc124d-3215-49d9-8afa-c4a2ccf6a066'

api.login(YOUR_ACCESS_TOKEN)

# Try to upload file
print(f"Uploading {file_path} to {repo_id}...")
print(f"Target path: {target_path}")

# Check if file exists
if not os.path.exists(file_path):
    print(f"Error: File {file_path} does not exist!")
    exit(1)

file_size = os.path.getsize(file_path)
print(f"File size: {file_size / 1024 / 1024:.2f} MB")

# Upload file
try:
    api.upload_file(
        repo_id=repo_id,
        path_or_fileobj=file_path,
        path_in_repo=target_path,
        commit_message=f"Upload {target_path}",
    )
    print("Upload successful!")
except Exception as e:
    print(f"Error during upload: {e}")
    raise
