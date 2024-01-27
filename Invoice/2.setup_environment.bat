conda create -n invoice python=3.10 -y
conda activate invoice
conda install -y cudatoolkit -c conda-forge
conda install -y cudnn -c nvidia
conda install -y cuda -c nvidia
conda install -y -c conda-forge ffmpeg libsndfile
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install TTS
pip install pyannote.audio
pip install librosa
pip install pydub
pip install moviepy
pip install pytube
pip install spleeter
pip install soundstretch
pip install accelerate
pip install deep-translator
pip install transformers==4.36.2 --force-reinstall
pip install ffmpeg-python
pip install numba
pip install language-tool-python
pip install protobuf==3.20 --force-reinstall