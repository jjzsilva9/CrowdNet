#!/usr/bin/env bash
# Setup script for DressCode PBR texture generation (submodule + weights + conda env).
# Run from CrowdNet repo root: bash setup_dresscode.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
MATGEN_DIR="$REPO_ROOT/external/DressCode/nn/material_gen"

# 1. Init submodule if needed
echo "=== Initialising DressCode submodule ==="
git submodule update --init external/DressCode

# 2. Download model weights from HuggingFace
echo ""
echo "=== Downloading material-gen weights from HuggingFace ==="
WEIGHTS=(
    "material_gen/unet/diffusion_pytorch_model.bin"
    "material_gen/text_encoder/pytorch_model.bin"
    "material_gen/vae/diffusion_pytorch_model.bin"
    "material_gen/refine_vae/vae_checkpoint_diffuse/diffusion_pytorch_model.safetensors"
    "material_gen/refine_vae/vae_checkpoint_normal/diffusion_pytorch_model.safetensors"
    "material_gen/refine_vae/vae_checkpoint_roughness/diffusion_pytorch_model.safetensors"
    "material_gen/refine_vae/vae_checkpoint_diffuse/config.json"
    "material_gen/refine_vae/vae_checkpoint_normal/config.json"
    "material_gen/refine_vae/vae_checkpoint_roughness/config.json"
)

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

hf download IHe-KaiI/DressCode "${WEIGHTS[@]}" --local-dir "$TMPDIR" 2>&1 | tail -5

echo "Copying weights into submodule tree..."
for W in "${WEIGHTS[@]}"; do
    SRC="$TMPDIR/$W"
    DST="$REPO_ROOT/external/DressCode/nn/$W"
    if [ -f "$SRC" ]; then
        cp "$SRC" "$DST"
        echo "  $W"
    fi
done

# 3. Create conda env if it doesn't exist
echo ""
echo "=== Setting up dresscode conda environment ==="
if conda info --envs | grep -q dresscode; then
    echo "Conda env 'dresscode' already exists, skipping creation."
else
    conda create -n dresscode python=3.9 -y
fi

echo "Installing PyTorch + dependencies..."
eval "$(conda shell.bash hook)"
conda activate dresscode

pip install torch==2.0.0+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install 'diffusers==0.24.0' 'transformers==4.35.2' 'huggingface_hub<0.24' accelerate
pip install xatlas trimesh numpy

# 4. Fix execstack if needed
TORCH_LIB="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib"
if [ -f "$REPO_ROOT/clear_exec.py" ]; then
    echo ""
    echo "=== Fixing execstack on torch libs ==="
    python "$REPO_ROOT/clear_exec.py" "$TORCH_LIB"/*.so
fi

# 5. WSL CUDA fix
echo ""
echo "=== Setting up WSL CUDA LD_LIBRARY_PATH ==="
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' > "$ACTIVATE_DIR/wsl_cuda.sh"

echo ""
echo "=== Done! ==="
echo "Activate with: conda activate dresscode"
echo "Test with:     python test_dresscode_texgen.py"
