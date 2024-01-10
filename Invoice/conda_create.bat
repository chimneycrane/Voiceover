@echo off
cd %~dp0
pushd %~dp0
conda update -n base -c defaults conda -y
conda config --add channels pytorch
conda config --add channels nvidia
conda config --add channels anaconda
conda create -n "invoice" python=3.10 -y
conda activate invoice
conda install -n invoice cudatoolkit -y
conda install -n invoice pytorch-cuda=12.1 -y
conda install -n invoice pytorch -y
conda install -n invoice torchvision -y
conda install -n invoice torchaudio -y
conda install -n invoice torchtext -y
conda config --set pip_interop_enabled True
pip install -r requirements.txt
conda update --all -y
conda invoice export > environment.yml