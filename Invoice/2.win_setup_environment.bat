conda create -n invoice python=3.10 -y
conda activate invoice
conda install -y cudatoolkit -c conda-forge
conda install -y cudnn -c nvidia
conda install -y cuda -c nvidia
conda install -y -c conda-forge ffmpeg libsndfile
pip install -r requirements.txt