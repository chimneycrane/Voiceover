cd %~dp0
pushd %~dp0
winget install -e --id Anaconda.Miniconda3
winget install -e --id Nvidia.CUDA
conda update -n base conda
conda update --all