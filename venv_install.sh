# conda create -y -n hislam python=3.10
# conda activate hislam
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirement.txt
conda install -y opencv suitesparse

python setup.py install

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/