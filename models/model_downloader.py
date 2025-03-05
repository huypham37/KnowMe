from huggingface_hub import snapshot_download

# Download the complete model
model_path = snapshot_download(
    repo_id="selfrag/selfrag_llama2_7b",  
    local_dir="/Users/mac/mlx-model/selfrag_llama2_7b_mlx",
    local_dir_use_symlinks=False  # Set to True to save space with symlinks
)

print(f"Model downloaded to: {model_path}")