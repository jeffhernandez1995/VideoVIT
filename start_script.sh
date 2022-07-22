# Setup Ubuntu
sudo apt update --yes
sudo apt upgrade --yes

# Get Miniconda and make it the main Python interpreter
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda 
rm ~/miniconda.sh

export PATH=~/miniconda/bin:$PATH

conda activate base

conda install -y mamba -n base -c conda-forge

mamba create -y -n torch python=3.8 gcc=9.4 cupy pkg-config compilers libjpeg-turbo opencv numba ffmpeg av -c conda-forge
mamba install -y -n torch pytorch torchvision cudatoolkit=11.3 torchaudio  -c pytorch
mamba install -y -n torch -c conda-forge cudatoolkit-dev=11.3

conda activate torch
pip install ffcv wandb
mamba remove -y torchvision
git clone https://github.com/pytorch/vision.git
cd vision
FORCE_CUDA=1 TORCHVISION_USE_FFMPEG=1 python setup.py install
cd ..
