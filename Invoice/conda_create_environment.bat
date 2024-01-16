conda config --add channels conda-forge --add channels pytorch --add channels nvidia
conda config --set pip_interop_enabled True
conda env create -f environment.yml
conda activate invoice
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
