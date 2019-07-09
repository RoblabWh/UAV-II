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

#certificates required by tensorrt
RUN apt-get install -y ca-certificates wget

#install tensorrt
ADD ./nv-tensorrt-repo-ubuntu1804-cuda10.0-trt5.0.2.6-ga-20181009_1-1_amd64.deb ./nv-tensorrt-repo-ubuntu1804-cuda10.0-trt5.0.2.6-ga-20181009_1-1_amd64.deb
RUN apt-get install ./nv-tensorrt-repo-ubuntu1804-cuda10.0-trt5.0.2.6-ga-20181009_1-1_amd64.deb
RUN apt-key add /var/nv-tensorrt-repo-cuda10.0-trt5.0.2.6-ga-20181009/7fa2af80.pub
RUN apt-get update

RUN apt-mark hold libcudnn7
RUN apt-mark hold cuda*
RUN apt-get install -y nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda10.0
RUN apt-get update
RUN apt-get install -y --no-install-recommends libnvinfer5=5.0.2-1+cuda10.0
RUN apt-get install -y --no-install-recommends libnvinfer-dev=5.0.2-1+cuda10.0 
RUN apt-get install -y tensorrt
RUN apt-get install -y python3-libnvinfer-dev
RUN apt-get install -y uff-converter-tf

#install utilies  
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
            wget \
            curl \
            ffmpeg less \
            nano

RUN apt-get update && apt-get install -y swig python3-tk mc locales

#install python and pip 
RUN apt-get update && apt-get install -y python3-dev python3-pip


#Setup pip
RUN pip3 --no-cache-dir install --upgrade \
    pip \
    setuptools

#install common python modules
RUN pip --no-cache-dir install \
    Pillow \
    matplotlib \
    mock \
    numpy \
    scipy \
    sklearn \
    pandas \
    opencv-python \
    h5py \
    keras_applications \
    keras_preprocessing \
    tensorflow-gpu \
    'pycuda>=2017.1.1'
    

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

RUN cd ~
RUN sudo locale-gen en_US
#RUN sudo chown -R ${USERNAME} ~/.*
#RUN sudo chown -R ${USERNAME} ~/*
