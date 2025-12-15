#!/bin/bash

# =============================================================================
# Miles Build Script (CUDA 12.8 Slurm Version)
# 
# This script uses uv for Python environment management.
# It relies on the Slurm module system for CUDA 12.8 instead of pip packages.
# 
# Configuration:
# - CUDA: 12.8 (via module load)
# - PyTorch: 2.8.0 (cu128)
# - Flash Attention 3: Prebuilt for cu128 + torch2.8
# - Flash Attention 2: Prebuilt for cu128 + torch2.8
# =============================================================================

set -e  # Exit on error

BASE_DIR="$(pwd)" # change this to your base directory

if [ -z "$BASE_DIR" ]; then
    echo "BASE_DIR is not set. Please set it to proceed with the installation."
    exit 1
fi

# =============================================================================
# Load Slurm Module
# =============================================================================
echo "Loading CUDA 12.8 module..."
module load CUDA/12.8

# Verify NVCC is in path
if ! command -v nvcc &> /dev/null; then
    echo "CRITICAL ERROR: nvcc not found after loading module."
    echo "Please ensure 'module load CUDA/12.8' works on this cluster."
    exit 1
fi

# =============================================================================
# Install uv if not already installed
# =============================================================================
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# =============================================================================
# Create Python virtual environment with uv
# =============================================================================
cd "$BASE_DIR"

# Create virtual environment with Python 3.12
uv venv --python 3.12 miles-venv

# Activate the virtual environment
source "$BASE_DIR/miles-venv/bin/activate"

# =============================================================================
# Install PyTorch with CUDA 12.8
# =============================================================================
echo "Installing PyTorch 2.8.0 with CUDA 12.8..."

# Install cuda-python (Pinned to 12.8 to match the module)
uv pip install cuda-python==12.8.0

# Install PyTorch 2.8.0 for CUDA 12.8
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Set TORCH_CUDA_ARCH_LIST for our GPU architectures
# 8.0 = A100 (Ampere), 9.0 = H100 (Hopper)
export TORCH_CUDA_ARCH_LIST="8.0;9.0"

# =============================================================================
# Install sglang
# =============================================================================
echo "Installing sglang..."
cd "$BASE_DIR"
if [ ! -d "$BASE_DIR/sglang" ]; then
    git clone https://github.com/sgl-project/sglang.git
fi
cd sglang
git checkout 303cc957e62384044dfa8e52d7d8af8abe12f0ac
uv pip install -e "python[all]"

# =============================================================================
# Install build tools
# =============================================================================
uv pip install cmake ninja packaging build wheel

# =============================================================================
# Install Flash Attention 3 (prebuilt wheels for cu128 + torch2.8)
# =============================================================================
echo "Installing Flash Attention 3..."
# Using windreamer's wheel index for CUDA 12.8 and PyTorch 2.8.0
uv pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280 --extra-index-url https://download.pytorch.org/whl/cu128

# =============================================================================
# Install Flash Attention 2 (prebuilt wheel for Megatron compatibility)
# =============================================================================
echo "Installing Flash Attention 2..."
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.18/flash_attn-2.7.4%2Bcu128torch2.8-cp312-cp312-linux_x86_64.whl

# =============================================================================
# Install mbridge, transformer_engine, flash-linear-attention
# =============================================================================
echo "Installing mbridge, transformer_engine, flash-linear-attention..."
uv pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps

# Transformer Engine 2.8.0 (compatible with CUDA 12.8)
uv pip install --no-build-isolation "transformer_engine[pytorch]==2.8.0" --no-cache-dir

uv pip install flash-linear-attention==0.4.0

# =============================================================================
# Install NVIDIA Apex (requires CUDA compilation)
# =============================================================================
echo "Installing NVIDIA Apex (Compiling from source)..."
# We utilize the loaded CUDA module for compilation
NVCC_APPEND_FLAGS="--threads 4" \
    APEX_CPP_EXT=1 \
    APEX_CUDA_EXT=1 \
    APEX_PARALLEL_BUILD=8 \
    uv pip install -v --no-cache-dir \
        --no-build-isolation \
        git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

# =============================================================================
# Install Megatron-LM
# =============================================================================
echo "Installing Megatron-LM..."
cd "$BASE_DIR"
if [ ! -d "$BASE_DIR/Megatron-LM" ]; then
    git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi
cd Megatron-LM
git checkout core_v0.14.0
uv pip install -e .

# =============================================================================
# Install additional dependencies
# =============================================================================
echo "Installing additional dependencies..."
uv pip install poetry pybind11
uv pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@9b8b788fdeb9c2ee528183214cef65a99b71e7d5 --no-cache-dir --force-reinstall
uv pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
uv pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

# =============================================================================
# Install remaining packages
# =============================================================================
uv pip install sglang_router ring_flash_attn pylatexenc
uv pip install -U "ray[data,train,tune,serve]"

# =============================================================================
# Install miles
# =============================================================================
echo "Installing miles..."
if [ ! -d "$BASE_DIR/miles" ]; then
    cd "$BASE_DIR"
    git clone https://github.com/radixark/miles.git
    cd miles/
    export MILES_DIR="$BASE_DIR/miles"
    uv pip install -e .
elif [ -f "$BASE_DIR/pyproject.toml" ]; then
    export MILES_DIR="$BASE_DIR"
    cd "$MILES_DIR"
    uv pip install -e .
else
    export MILES_DIR="$BASE_DIR/miles"
    cd "$MILES_DIR"
    uv pip install -e .
fi

# =============================================================================
# Apply patches
# =============================================================================
echo "Applying patches..."
cd "$BASE_DIR/sglang"
git apply "$MILES_DIR/docker/patch/v0.5.5.post1/sglang.patch" || echo "sglang patch already applied or failed"

cd "$BASE_DIR/Megatron-LM"
git apply "$MILES_DIR/docker/patch/v0.5.5.post1/megatron.patch" || echo "Megatron patch already applied or failed"

echo ""
echo "============================================================================="
echo "Installation complete!"
echo ""
echo "To activate the environment, run:"
echo "  module load CUDA/12.8"
echo "  source $BASE_DIR/miles-venv/bin/activate"
echo ""
echo "Environment configured using System CUDA from 'module load CUDA/12.8'"
echo "PyTorch Version: 2.8.0 (cu128)"
echo "CUDA_HOME: $CUDA_HOME"
echo "============================================================================="