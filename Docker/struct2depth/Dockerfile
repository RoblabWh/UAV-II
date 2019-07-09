FROM nvidia/cuda:10.0-devel
MAINTAINER roblabfhge <roblab2011@googlemail.com>

#ENV CUDNN_VERSION 7.4.2.24
ENV CUDNN_VERSION 7.5.0.56
ENV DEBIAN_FRONTEND=newt
ENV DEBCONF_NONINTERACTIVE_SEEN=true

#ARG UBUNTU_VERSION=16.04
#ARG ARCH=
#ARG CUDA=9.0

#FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
#LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

#ENV CUDNN_VERSION 7.4.2.24
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda10.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda10.0 && \
	    apt-mark hold libcudnn7

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libxau-dev \
        libxdmcp-dev \
        libxcb1-dev \
        libxext-dev \
        libx11-dev 

# replace with other Ubuntu version if desired
# see: https://hub.docker.com/r/nvidia/opengl/
COPY --from=nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04 \
  /usr/local/lib/x86_64-linux-gnu \
  /usr/local/lib/x86_64-linux-gnu

# replace with other Ubuntu version if desired
# see: https://hub.docker.com/r/nvidia/opengl/
COPY --from=nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04 \
  /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json \
  /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json

RUN echo '/usr/local/lib/x86_64-linux-gnu' >> /etc/ld.so.conf.d/glvnd.conf && \
    ldconfig && \
    echo '/usr/local/$LIB/libGL.so.1' >> /etc/ld.so.preload && \
    echo '/usr/local/$LIB/libEGL.so.1' >> /etc/ld.so.preload

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
    
RUN apt-get update && apt-get install -y --no-install-recommends \
    	    mesa-utils sudo module-init-tools \
	    emacs git build-essential\
	    libcurl3-dev \
	    libfreetype6-dev \
            libhdf5-serial-dev \
            libzmq3-dev \
            pkg-config \
            rsync \
            software-properties-common \
            unzip \
            zip \
            zlib1g-dev \
            wget python3-pip

RUN apt-get update && apt-get install -y \
    curl \
    openjdk-8-jdk \
    python3-dev \
    swig python3-tk mc locales \
    ffmpeg less


RUN pip3 --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN pip --no-cache-dir install \
    Pillow \
    h5py \
    keras_applications \
    keras_preprocessing \
    matplotlib \
    mock \
    numpy \
    scipy \
    sklearn \
    pandas \
    enum34 \
    opencv-python 
 

RUN pip --no-cache-dir install \
    tensorflow-gpu

# Install bazel
ARG BAZEL_VERSION=0.19.2
RUN mkdir /bazel && \
    wget -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget -O /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh && \
    rm -f /bazel/installer.sh

# Define build arguments
ARG USER_ID
ARG GROUP_ID
ARG USERNAME

# Setup user
RUN mkdir -p /home/${USERNAME}
RUN echo "${USERNAME}:x:${USER_ID}:${GROUP_ID}:${USERNAME},,,:/home/${USERNAME}:/bin/bash" >> /etc/passwd
RUN echo "${USERNAME}:x:${USER_ID}:" >> /etc/group
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USERNAME}
RUN chmod 0440 /etc/sudoers.d/${USERNAME}
RUN chown ${USER_ID}:${GROUP_ID} -R /home/${USERNAME}

USER $USERNAME
ENV HOME /home/${USERNAME}

RUN cd ~ && git clone https://github.com/tensorflow/models

RUN sudo locale-gen en_US
RUN sudo locale-gen en_US.UTF-8
RUN sudo chown -R ${USERNAME} ~/.*
RUN sudo chown -R ${USERNAME} ~/*
