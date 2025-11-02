# export CUDA_HOME=$CONDA_PREFIX
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# bash /mnt/home/code/anaconda.sh -b
# # source ~/anaconda3/bin/activate
# eval "$($HOME/anaconda3/bin/conda shell.bash hook)"

# conda create -y -n superslam python=3.9
# conda activate superslam

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit

pip install -r requirement.txt

# RaDe-GS rasterizer
pip install thirdparty/diff-gaussian-rasterization
pip install thirdparty/simple-knn/

# CUT3R
conda install 'llvm-openmp<16'# for training logging
# for evaluation
# pip install evo
# pip install open3d
# Compile the cuda kernels for RoPE (as in CroCo v2)
cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../

# Lietorch
cd thirdparty/lietorch
python setup.py install
cd ../../

# GCC update
# sudo apt update
# sudo apt install gcc-11 g++-11

# sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
# sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110
# sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
# sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110

# sudo update-alternatives --config gcc
# sudo update-alternatives --config g++