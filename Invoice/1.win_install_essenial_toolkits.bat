cd %~dp0
pushd %~dp0
winget install -e --id RProject.R -v 4.2.2 --silent
setx R_HOME "C:\Program Files\R\R-4.2.2\bin" /M
setx PATH "%PATH%;C:\Program Files\R\R-4.2.2\bin"
winget install -e --id Anaconda.Miniconda3
winget install -e --id Nvidia.CUDA
conda update -n base conda
conda update --all