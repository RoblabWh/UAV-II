FROM ubuntu:16.04

# Install some basic tools
RUN apt-get update && apt-get install -y \
    sudo \
    gosu \
    ninja-build \
    build-essential \
    moreutils \
    software-properties-common \
    nano \
    vim \
    emacs \
    curl 

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libxau-dev \
        libxdmcp-dev \
        libxcb1-dev \
        libxext-dev \
        libx11-dev \
        git \
        cmake \
        libeigen3-dev

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

# install dependencies
RUN sudo apt-get install -y \
    libglew-dev \
    libopencv-dev
# build + install pangolin
RUN cd /tmp/ && git clone https://github.com/stevenlovegrove/Pangolin.git  && cd Pangolin && git checkout v0.5 && mkdir build && cd build && cmake .. && cmake --build . && make && make install && cd ../.. && rm -rf Pangolin 

# Define build arguments
ARG USER_ID
ARG GROUP_ID
ARG USERNAME

# Setup user
RUN mkdir -p /home/${USERNAME} && \
    echo "${USERNAME}:x:${USER_ID}:${GROUP_ID}:${USERNAME},,,:/home/${USERNAME}:/bin/bash" >> /etc/passwd && \
    echo "${USERNAME}:x:${USER_ID}:" >> /etc/group && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME} && \
    chown ${USER_ID}:${GROUP_ID} -R /home/${USERNAME}

WORKDIR /home/${USERNAME}

USER $USERNAME
ENV HOME /home/${USERNAME}

RUN sudo chown -R ${USERNAME} ~/.*




