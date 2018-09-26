FROM nvidia/cuda:8.0-cudnn6-devel

# Install curl and sudo
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
 && rm -rf /var/lib/apt/lists/*

# Use Tini as the init process with PID 1
RUN curl -Lso /tini https://github.com/krallin/tini/releases/download/v0.14.0/tini \
 && chmod +x /tini
ENTRYPOINT ["/tini", "--"]

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# Install Git, bzip2, and X11 client
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    git \
    bzip2 \
    libx11-6 \
 && sudo rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.3.14-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name pytorch-py36 \
    python=3.6 numpy pyyaml scipy ipython mkl \
 && /home/user/miniconda/bin/conda clean -ya
ENV PATH=/home/user/miniconda/envs/pytorch-py36/bin:$PATH \
    CONDA_DEFAULT_ENV=pytorch-py36 \
    CONDA_PREFIX=/home/user/miniconda/envs/pytorch-py36

# CUDA 8.0-specific steps
RUN conda install -y --name pytorch-py36 -c soumith \
    magma-cuda80 \
 && conda clean -ya

# Install PyTorch and Torchvision
RUN conda install -y --name pytorch-py36 -c soumith \
    pytorch=0.2.0 torchvision=0.1.9 \
 && conda clean -ya

# Install HDF5 Python bindings
RUN conda install -y --name pytorch-py36 \
    h5py \
 && conda clean -ya
RUN pip install h5py-cache

# Install Requests, a Python library for making HTTP requests
RUN conda install -y --name pytorch-py36 requests && conda clean -ya

# Install Graphviz
RUN conda install -y --name pytorch-py36 graphviz=2.38.0 \
 && conda clean -ya
RUN pip install graphviz

# Install django cors headers
RUN pip install django-cors-headers

# Add to crontab
RUN sudo apt-get update && sudo apt-get -y install cron
# echo new cron into cron file
RUN sudo echo "0 * * * * sh /app/fast_neural_style/start_synchronizeModels.sh  >> /app/fast_neural_style/django/synchronizeModels.log 2>&1" >> synchronizeModel
# install new cron file
RUN sudo crontab synchronizeModel
# start cron
RUN sudo /etc/init.d/cron start

# Install from requirements.txt
RUN pip install -r requirements.txt

# Expose port
# Four backend subservices
EXPOSE 30006
EXPOSE 33100
EXPOSE 31004
EXPOSE 35010
# Distributor service
EXPOSE 36060

# Change the owner of django folder
RUN sudo chown user fast_neural_style/django


# Set the default command to python3
CMD ["./fast_neural_style/start_django.sh"]
