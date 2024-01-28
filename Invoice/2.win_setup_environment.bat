conda create -n invoice python=3.10 -y
conda activate invoice
conda install -y cudatoolkit -c conda-forge
conda install -y cudnn -c nvidia
conda install -y cuda -c nvidia
conda install -y -c conda-forge ffmpeg libsndfile
pip install -r requirements.txt
Rscript -e "install.packages("pbapply", repos = "https://cran.rstudio.com", quiet = TRUE)"
Rscript -e "install.packages("tuneR", repos = "https://cran.rstudio.com", quiet = TRUE)"
Rscript -e "install.packages("seewave", repos = "https://cran.rstudio.com", quiet = TRUE)"
Rscript -e "install.packages("fftw", repos = "https://r-forge.r-project.org", quiet = TRUE)"