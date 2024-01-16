@echo off
cd %~dp0
pushd %~dp0
winget install -e --id Gyan.FFmpeg
winget install -e --id Anaconda.Miniconda3
winget install -e --id Nvidia.CUDA