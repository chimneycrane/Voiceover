conda create -n invoice python=3.10 -y
conda activate invoice
conda config --add channels conda-forge --add channels pytorch --add channels nvidia
conda update -n base -c defaults conda -y
python.exe -m pip install --upgrade pip
pip install wheel --upgrade
#pip install tensorflow-gpu
conda config --set pip_interop_enabled True
conda install -n invoice -y cmake conda-forge::blas=*=openblas
conda install -n invoice -y scikit-build pybind11
conda install -n invoice -y numpy scipy nomkl
pip install speechbrain
pip install TTS
pip install pyannote.audio
pip install librosa
pip install pydub
pip install moviepy
pip install pytube
pip install spleeter
pip install audiostretchy
pip install accelerate
pip install deep-translator
conda install -n invoice -y cudatoolkit
conda install -n invoice -y cudnn
conda install -n invoice -y cuda
conda install -n invoice -y -c conda-forge ffmpeg libsndfile
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
conda env export > environment.yml