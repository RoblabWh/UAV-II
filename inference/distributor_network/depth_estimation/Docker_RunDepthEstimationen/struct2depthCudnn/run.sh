#!/bin/bash

# if [ $# -ne 1 ]; then
#   exit 1
# fi
echo "cp /home/gas/host/docker/struct2depth/util.py ~/models/research/struct2depthcudnn"
# im_data = np.fromstring(gfile.Open(img_file,'rb').read(), np.uint8) // add 'rb' line 61

xhost +local:root

nvidia-docker run -it --rm \
	      --name struct2depthcudnn \
	      --privileged \
	      --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	      --volume="/opt/clion:/opt/clion" \
	      --volume="/home/${USER}:/home/${USER}/host" \
	      --volume="/media/${USER}/DATA:/home/${USER}/DATA" \
	      --env="XAUTHORITY=$XAUTH" \
	      --env="DISPLAY=$DISPLAY" \
	      --env="QT_X11_NO_MITSHM=1" \
	      --device="/dev/video0:/dev/video0" \
	      --device="/dev/dri:/dev/dri" \
	      --net=host \
	      struct2depthcudnn:latest \
              /bin/bash
xhost -local:root
