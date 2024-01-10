@echo off
cd %~dp0
pushd %~dp0
conda install --n invoice -file requirements.txt -y