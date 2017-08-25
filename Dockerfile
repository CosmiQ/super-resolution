FROM cosmiqworks/spacenet-utilities-gpu

MAINTAINER Patrick Hagerty

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        vim \
        wget \ 
        eog \
        zip \
        zlib1g-dev \
        libopencv-dev \
        python-opencv \
        build-essential autoconf libtool libcunit1-dev \
        libproj-dev libgdal-dev libgeos-dev libjson0-dev vim python-gdal \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        tensorflow-gpu \
        && \
    python -m ipykernel.kernelspec

ENV GIT_BASE=/opt/
WORKDIR $GIT_BASE
# Set up our notebook config.
RUN git clone --depth 1 https://github.com/CosmiQ/super-resolution.git

WORKDIR /workspace

# TensorBoard
RUN ["/bin/bash"]
