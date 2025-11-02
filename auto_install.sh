#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# -----------------------------
# 1. Install Anaconda (silent mode)
# -----------------------------
ANACONDA_INSTALLER="/mnt/home/code/anaconda.sh"
INSTALL_DIR="$HOME/anaconda3"

if [ ! -d "$INSTALL_DIR" ]; then
    echo "Installing Anaconda..."
    bash "$ANACONDA_INSTALLER" -b -p "$INSTALL_DIR"
else
    echo "Anaconda already installed at $INSTALL_DIR"
fi

# -----------------------------
# 2. Initialize conda
# -----------------------------
echo "Initializing conda..."
eval "$($INSTALL_DIR/bin/conda shell.bash hook)"

# -----------------------------
# 3. Create superslam environment
# -----------------------------
ENV_NAME="superslam"
PYTHON_VERSION="3.9"

if conda env list | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

# -----------------------------
# 4. Activate environment
# -----------------------------
conda activate "$ENV_NAME"

# -----------------------------
# 5. Install PyTorch + CUDA
# -----------------------------
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing CUDA toolkit..."
conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit

# -----------------------------
# 6. Install requirements.txt
# -----------------------------
if [ -f "requirement.txt" ]; then
    echo "Installing Python dependencies from requirement.txt..."
    pip install -r requirement.txt
else
    echo "requirement.txt not found. Skipping."
fi

# -----------------------------
# 7. Install RaDe-GS rasterizer
# -----------------------------
if [ -d "thirdparty/diff-gaussian-rasterization" ]; then
    pip install thirdparty/diff-gaussian-rasterization
fi
if [ -d "thirdparty/simple-knn" ]; then
    pip install thirdparty/simple-knn/
fi

# -----------------------------
# 8. Install CUT3R dependencies
# -----------------------------
echo "Installing CUT3R dependencies..."
conda install -y 'llvm-openmp<16'  # for training logging
pip install evo
pip install open3d

# Compile RoPE CUDA kernels
if [ -d "src/croco/models/curope" ]; then
    echo "Compiling RoPE CUDA kernels..."
    cd src/croco/models/curope/
    python setup.py build_ext --inplace
    cd ../../../../
fi

# -----------------------------
# 9. Install Lietorch
# -----------------------------
if [ -d "thirdparty/lietorch" ]; then
    echo "Installing Lietorch..."
    cd thirdparty/lietorch
    python setup.py install
    cd ../../
fi

echo "---------------------------------"
echo "All done! Conda environment '$ENV_NAME' is ready."
