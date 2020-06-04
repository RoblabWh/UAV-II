#!/bin/bash

xhost +local:root

QT_GRAPHICSSYSTEM="native" nvidia-docker run -it --rm \
	      --name densedepth_roscudnn \
	      --privileged \
	      --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	      --volume="/opt/clion:/opt/clion" \
	      --volume="/home/marc:/home/${USER}/host" \
              --volume="/home/marc/Dockerfiles/densedepth_ROSCudnn/DepthInferenceAndVisualization:/home/marc/DepthInferenceAndVisualization" \
	      --volume="/media/marc/DATA:/home/marc/DATA" \
	      --env="XAUTHORITY=$XAUTH" \
	      --env="DISPLAY=$DISPLAY" \
	      --env="QT_X11_NO_MITSHM=1" \
	      --device="/dev/video0:/dev/video0" \
	      --device="/dev/dri:/dev/dri" \
	      --net=host \
	      densedepth_roscudnn:latest \
              /bin/bash
xhost -local:root
