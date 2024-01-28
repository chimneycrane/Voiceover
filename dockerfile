# Base image with CUDA toolkit and drivers
FROM nvidia/cuda:11.8-base-ubuntu20.04

# Install essential system packages
RUN apt-get update && apt-get install -y \
    nvidia-cuda-toolkit \
    libsndfile1 \
    python3.10 \
    python3-pip \
    git \
    r-base \  # Install R
    libcurl4-openssl-dev  # Dependency for Rscript
RUN pip install --upgrade pip
RUN pip install wheel

#set R_HOME for R
ENV R_HOME /usr/lib/R
ENV PATH="${R_HOME}/bin:${PATH}"

# Install R packages
RUN Rscript -e "install.packages('pbapply', repos = 'https://cran.rstudio.com', quiet = TRUE)"
RUN Rscript -e "install.packages('tuneR', repos = 'https://cran.rstudio.com', quiet = TRUE)"
RUN Rscript -e "install.packages('seewave', repos = 'https://cran.rstudio.com', quiet = TRUE)"
RUN Rscript -e "install.packages('fftw', repos = 'https://cran.rstudio.com', quiet = TRUE)"

RUN mkdir /shared
VOLUME Docker_shared:/shared
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install git+https://github.com/chimneycrane/Voiceover.git