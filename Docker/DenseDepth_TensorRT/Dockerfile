FROM tensorrt:latest
MAINTAINER roblabfhge <roblab2011@googlemail.com>
USER root

RUN pip --no-cache-dir uninstall -y pySide2
RUN apt-get update
RUN pip --no-cache-dir install PyQt5
RUN pip --no-cache-dir install numpy
RUN pip --no-cache-dir install imageio
RUN pip --no-cache-dir install scipy
RUN pip --no-cache-dir install 'pycuda>=2017.1.1'

#ADD ./DenseDepth ./DenseDepth
#ADD ./KerasToTensorRT-master ./KerasToTensorRT-master

ARG USERNAME
USER $USERNAME
ENV HOME /home/${USERNAME}
