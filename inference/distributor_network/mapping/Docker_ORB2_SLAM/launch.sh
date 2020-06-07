#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
echo $parent_path

xhost + local:
QT_GRAPHICSSYSTEM="native" nvidia-docker run --entrypoint "/bin/bash" -it --rm \
	--name container_orb \
    --privileged \
   	-e DISPLAY=unix$DISPLAY \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /etc/machine-id:/etc/machine-id \
    -v "$parent_path/../config:/home/${USER}/ORB_SLAM2/config" \
    -v "$parent_path/../home:/home/${USER}" \
    -v /home/${USER}:/home/${USER}/host \
    -v /dev/bus/usb:/dev/bus/usb \
    --device="/dev/dri:/dev/dri" \
	--network="host" \
	--ipc="host" \
	uav-ii/orb \
	

xhost -local:
