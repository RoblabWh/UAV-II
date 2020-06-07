FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

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
        python3 \
        python3-setuptools \
        python3-pip
        
RUN pip3 install --upgrade pip
        
RUN pip3 install \
PySide2~=5.14.1 \
Pillow~=6.2.1 \
numpy~=1.17.2 \
opencv-python~=4.1.1.26 \
rsa~=4.0 \
pyasn1~=0.4.7 \
vtk~=8.1.2 \
pyzmq~=18.1.0 \
pexpect~=4.7.0 \
tornado~=6.0.3 \
six~=1.12.0 \
gast~=0.2.2 \
h5py~=2.10.0 \
ipython~=7.8.0 \
jedi~=0.15.1 \
parso~=0.5.1 \
nose~=1.3.7 \
setuptools~=41.6.0 \
astor~=0.8.0 \
numba~=0.49.1 \
llvmlite~=0.32.1 \
Pygments~=2.4.2 \
scipy~=1.4.1 \
Jinja2~=2.10.3 \
ipykernel~=5.1.3 \
nbformat~=4.4.0 \
matplotlib~=3.1.1 \
wheel~=0.33.6 \
Cython~=0.29.14 \
webencodings~=0.5.1 \
bleach~=3.1.0 \
cachetools~=3.1.1 \
certifi~=2019.9.11 \
urllib3~=1.25.7 \
requests~=2.22.0 \
MarkupSafe~=1.1.1 \
open3d~=0.8.0.0 \
ipywidgets~=7.5.1 \
traitlets~=4.3.3 \
testpath~=0.4.2 \
backcall~=0.1.0 \
decorator~=4.4.0 \
pickleshare~=0.7.5 \
wcwidth~=0.1.7 \
shiboken2~=5.14.1 \
chardet~=3.0.4 \
ptyprocess~=0.6.0 \
tensorflow-gpu~=1.15.3 \
python-dateutil~=2.8.0 \
Markdown~=3.1.1 \
jsonschema~=3.1.1 \
notebook~=6.0.1 \
Send2Trash~=1.5.0 \
terminado~=0.8.2 \
oauthlib~=3.1.0 \
idna~=2.8 \
Werkzeug~=0.16.0 \
termcolor~=1.1.0 \
nbconvert~=5.6.0 \
defusedxml~=0.6.0 \
pandocfilters~=1.4.2 \
mistune~=0.8.4 \
entrypoints~=0.3 \
attrs~=19.3.0 \
pyrsistent~=0.15.4 \
cycler~=0.10.0 \
pyparsing~=2.4.2 \
kiwisolver~=1.1.0 \
tensorboard~=1.15.0 \
pyquaternion~=0.9.5 \
wrapt~=1.11.2 \
zipp~=0.6.0 \
pptk~=0.1.0

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




ENV HOME /home/${USERNAME}

USER $USERNAME
WORKDIR /home/${USERNAME}

RUN sudo chown -R ${USERNAME} ~/.*
#RUN sudo chown -R ${USERNAME} ~/*




