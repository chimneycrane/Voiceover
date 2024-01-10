@echo off
cd %~dp0
pushd %~dp0
conda config --add channels pytorch
conda config --add channels nvidia
conda config --add channels anaconda
conda install -n invoice cudatoolkit -y
conda install -n invoice pytorch-cuda=12.1 -y
conda install -n invoice pytorch -y
conda install -n invoice torchvision -y
conda install -n invoice torchaudio -y
conda install --force-reinstall -y -q -n invoice --file requirements.txt