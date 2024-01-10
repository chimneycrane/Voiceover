@echo off
cd %~dp0
pushd %~dp0
conda activate invoice
python transcribe.py %2 %3 %4 %5