FROM densedepth:latest
MAINTAINER roblabfhge <roblab2011@googlemail.com>
USER root

RUN apt-get update
RUN apt-get install -y ca-certificates wget
RUN apt-get install nano

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
RUN dpkg -l | grep TensorRT

#ADD ./DenseDepth ./DenseDepth
#ADD ./KerasToTensorRT-master ./KerasToTensorRT-master

ARG USERNAME
USER $USERNAME

ENV HOME /home/${USERNAME}
