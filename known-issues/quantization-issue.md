# Quantization Issues with Apple Silicon Mac

- **Type**: Issue
- **Priority**: High
- **Platform**: Apple Silicon Mac

## Description

bitsandbytes library is primarily designed for NVIDIA GPUs with CUDA support, which doesn't work properly on Apple Silicon Mac hardware.

## Primary Issues

1. **CUDA Dependency**: bitsandbytes relies on CUDA libraries not available for Mac
2. **Binary Compatibility**: The PyPI packages don't include proper binaries for Apple Silicon
3. **Quantization Failures**: Attempting to use `load_in_8bit=True` throws import errors

## Workaround Solution

Use Apple's MLX framework to convert and quantize Hugging Face models:

```bash
# Install required package
pip install mlx-lm

# Convert and quantize model to 8-bit precision
python -m mlx_lm.convert \
  --hf-path selfrag/selfrag_llama2_7b \
  --mlx-path ./selfrag_llama2_7b_mlx \
  -q \
  --q-bits 8
```

## Additional Issues and Solutions

### Issue 1: Format Incompatibility
MLX-LM requires safetensors format, but some Hugging Face models use .bin format

**Solution**: 
Use [Convert to Safetensors Utility](https://github.com/IBM/convert-to-safetensors) to convert bin files to safetensors

### Issue 2: Multi-Shard Model Compatibility
Convert to Safetensors Utility expects single shard models, but Hugging Face often provides multi-shard models

**Solution**: 
Merge shards into a single file using this script:

```python
import torch
import os

# Paths
source_dir = "[Source directory of the model]"  # Usually at ~/.cache/huggingface/hub/
output_file = os.path.join(source_dir, "pytorch_model.bin")

# Load the sharded model
model = torch.load(os.path.join(source_dir, "pytorch_model-00001-of-00002.bin"), map_location="cpu")
for i in range(2, 3):  # Adjust range if more shards exist
    shard_path = os.path.join(source_dir, f"pytorch_model-0000{i}-of-00002.bin")
    shard = torch.load(shard_path, map_location="cpu")
    model.update(shard)

# Save as a single .bin file
torch.save(model, output_file)

print(f"Merged weights saved to {output_file}")
```

## Alternative Approach

For simplicity, consider forcing CPU mode and avoiding quantization entirely by modifying model initialization code.