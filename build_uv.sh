#!/bin/bash

# =============================================================================
# Miles Build Script (uv-only version)
# 
# This script uses uv for Python environment management and pip-installable
# CUDA toolkit components instead of micromamba/conda
# 
# PyTorch wheels come bundled with CUDA runtime, so we only need to install
# the CUDA development tools (nvcc, headers, etc.) for packages that compile
# CUDA extensions (apex, transformer_engine, etc.)
# =============================================================================

set -e  # Exit on error

BASE_DIR=""

if [ -z "$BASE_DIR" ]; then
    echo "BASE_DIR is not set. Please set it to proceed with the installation."
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
# Install CUDA toolkit components via pip
# These provide nvcc and development headers needed for compiling CUDA extensions
# Using CUDA 12.9 to match PyTorch cu129 wheels
# =============================================================================
echo "Installing CUDA toolkit components via pip..."

# Core CUDA development tools
uv pip install nvidia-cuda-nvcc-cu12==12.9.86
uv pip install nvidia-cuda-runtime-cu12
uv pip install nvidia-cuda-cupti-cu12
uv pip install nvidia-cuda-nvrtc-cu12
uv pip install nvidia-cuda-cccl-cu12

# cuDNN and NCCL
uv pip install nvidia-cudnn-cu12
uv pip install nvidia-nccl-cu12

# NVTX for profiling (if needed)
uv pip install nvidia-nvtx-cu12

# =============================================================================
# Set up CUDA paths for packages installed via pip
# The nvidia packages install binaries and libraries to the site-packages
# =============================================================================
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
NVIDIA_PATH="$SITE_PACKAGES/nvidia"

export CUDA_HOME="$NVIDIA_PATH/cuda_nvcc"
export PATH="$NVIDIA_PATH/cuda_nvcc/bin:$PATH"
export LD_LIBRARY_PATH="$NVIDIA_PATH/cudnn/lib:$NVIDIA_PATH/nccl/lib:$NVIDIA_PATH/cuda_runtime/lib:$NVIDIA_PATH/cuda_cupti/lib:$NVIDIA_PATH/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
export CPATH="$NVIDIA_PATH/cuda_nvcc/include:$NVIDIA_PATH/cuda_runtime/include:$NVIDIA_PATH/cuda_cccl/include:$CPATH"

# Set TORCH_CUDA_ARCH_LIST for common modern GPU architectures
# 8.0 = A100 (Ampere), 8.6 = RTX 30xx (Ampere), 8.9 = RTX 40xx (Ada), 9.0 = H100 (Hopper)
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# =============================================================================
# Install PyTorch with CUDA 12.9 (bundled CUDA runtime)
# =============================================================================
echo "Installing PyTorch with CUDA 12.9..."
# Pin cuda-python to avoid installing cuda 13.0 for sglang
uv pip install cuda-python==12.9.1
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

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
# Install Flash Attention 3 (prebuilt wheels)
# =============================================================================
echo "Installing Flash Attention 3..."
cd "$BASE_DIR"
uv pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu129_torch280 --extra-index-url https://download.pytorch.org/whl/cu129

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
uv pip install --no-build-isolation "transformer_engine[pytorch]==2.8.0" --no-cache-dir
uv pip install flash-linear-attention==0.4.0

# =============================================================================
# Install NVIDIA Apex (requires CUDA compilation)
# =============================================================================
echo "Installing NVIDIA Apex..."
NVCC_APPEND_FLAGS="--threads 4" \
uv pip install --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
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
uv pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@9b8b788fdeb9c2ee528183214cef65a99b71e7d5 --no-cache-dir --force-reinstall
uv pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
uv pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

# =============================================================================
# Install remaining packages
# =============================================================================
uv pip install sglang_router
uv pip install ring_flash_attn
uv pip install -U "ray[data,train,tune,serve]"
uv pip install pylatexenc

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
echo "  source $BASE_DIR/miles-venv/bin/activate"
echo ""
echo "You may also need to set these environment variables after activation:"
echo "  SITE_PACKAGES=\$(python -c \"import site; print(site.getsitepackages()[0])\")"
echo "  NVIDIA_PATH=\"\$SITE_PACKAGES/nvidia\""
echo "  export CUDA_HOME=\"\$NVIDIA_PATH/cuda_nvcc\""
echo "  export PATH=\"\$NVIDIA_PATH/cuda_nvcc/bin:\$PATH\""
echo "  export LD_LIBRARY_PATH=\"\$NVIDIA_PATH/cudnn/lib:\$NVIDIA_PATH/nccl/lib:\$NVIDIA_PATH/cuda_runtime/lib:\$LD_LIBRARY_PATH\""
echo "  export TORCH_CUDA_ARCH_LIST=\"8.0;8.6;8.9;9.0\""
echo "============================================================================="

