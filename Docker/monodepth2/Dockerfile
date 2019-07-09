FROM struct2depth:latest
MAINTAINER roblabfhge <roblab2011@googlemail.com>

#ENV CUDNN_VERSION 7.4.2.24
ENV CUDNN_VERSION 7.5.0.56
ENV DEBIAN_FRONTEND=newt
ENV DEBCONF_NONINTERACTIVE_SEEN=true

#ARG UBUNTU_VERSION=16.04
#ARG ARCH=
#ARG CUDA=9.0

RUN sudo pip3 --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN sudo pip --no-cache-dir install \
    opencv-python==3.3.1.11 \
    torch \
    torchvision \
    tensorboardX


# Define build arguments
ARG USER_ID
ARG GROUP_ID
ARG USERNAME
USER $USERNAME

ENV HOME /home/${USERNAME}

