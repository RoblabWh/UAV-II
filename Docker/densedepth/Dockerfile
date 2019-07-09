FROM struct2depth:latest
MAINTAINER roblabfhge <roblab2011@googlemail.com>

RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
        build-essential \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-regex-dev \
        libboost-system-dev \
        libboost-test-dev \
        libeigen3-dev \
        libsuitesparse-dev \
        libfreeimage-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libcgal-qt5-dev

RUN sudo pip --no-cache-dir install \
    Pillow \
    h5py \
    matplotlib \
    mock \
    numpy \
    enum34 \
    keras \
    scikit-learn \
    scikit-image \
    opencv-python \
    pydot \
    GraphViz \
    PySide2 \
    PyOpenGL \
    PyOpenGL_accelerate \
    pyglm
 


# Define build arguments
ARG USER_ID
ARG GROUP_ID
ARG USERNAME



USER $USERNAME
ENV HOME /home/${USERNAME}
RUN cd ~ && mkdir xdg
ENV XDG_RUNTIME_DIR /home/xdg

RUN sudo usermod -a -G video ${USERNAME}
