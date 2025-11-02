#!/bin/bash
set -e  # 出现错误立即退出

# -----------------------------
# 1. 安装 Anaconda（静默模式）
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
# 2. 初始化 conda
# -----------------------------
echo "Initializing conda..."
eval "$($INSTALL_DIR/bin/conda shell.bash hook)"

# -----------------------------
# 3. 创建 superslam 环境
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
# 4. 激活环境
# -----------------------------
conda activate "$ENV_NAME"

# -----------------------------
# 5. 安装 PyTorch + CUDA
# -----------------------------
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing CUDA toolkit..."
conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit

# -----------------------------
# 6. 安装 requirements.txt
# -----------------------------
if [ -f "requirement.txt" ]; then
    echo "Installing Python dependencies from requirement.txt..."
    pip install -r requirement.txt
else
    echo "requirement.txt not found. Skipping."
fi

# -----------------------------
# 7. 安装 RaDe-GS rasterizer
# -----------------------------
if [ -d "thirdparty/diff-gaussian-rasterization" ]; then
    pip install thirdparty/diff-gaussian-rasterization
fi
if [ -d "thirdparty/simple-knn" ]; then
    pip install thirdparty/simple-knn/
fi

# -----------------------------
# 8. 安装 CUT3R 依赖
# -----------------------------
echo "Installing CUT3R dependencies..."
conda install -y 'llvm-openmp<16'  # for training logging
pip install evo
pip install open3d

# 编译 RoPE cuda kernels
if [ -d "src/croco/models/curope" ]; then
    echo "Compiling RoPE cuda kernels..."
    cd src/croco/models/curope/
    python setup.py build_ext --inplace
    cd ../../../../
fi

# -----------------------------
# 9. 安装 Lietorch
# -----------------------------
if [ -d "thirdpary/lietorch" ]; then
    echo "Installing Lietorch..."
    cd thirdpary/lietorch
    python setup.py install
    cd ../../
fi

echo "---------------------------------"
echo "All done! Conda environment '$ENV_NAME' is ready."
