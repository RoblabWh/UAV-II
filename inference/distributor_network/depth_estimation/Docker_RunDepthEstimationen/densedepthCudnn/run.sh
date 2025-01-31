#!/bin/bash

# if [ $# -ne 1 ]; then
#   exit 1
# fi
echo "python3 test_simple.py --image_path ~/host/docker/monodepth2/images --model_name mono_1024x320"
# im_data = np.fromstring(gfile.Open(img_file,'rb').read(), np.uint8) // add 'rb' line 61

xhost +local:root

QT_GRAPHICSSYSTEM="native" nvidia-docker run -it --rm \
	      --name densedepth \
	      --privileged \
	      --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	      --volume="/opt/clion:/opt/clion" \
	      --volume="/home/${USER}:/home/${USER}/host" \
              --volume="/home/${USER}/Dockerfiles/densedepth/DenseDepth:/home/${USER}/DenseDepth" \
	      --volume="/media/${USER}/DATA:/home/${USER}/DATA" \
	      --env="XAUTHORITY=$XAUTH" \
	      --env="DISPLAY=$DISPLAY" \
	      --env="QT_X11_NO_MITSHM=1" \
	      --device="/dev/video0:/dev/video0" \
	      --device="/dev/dri:/dev/dri" \
	      --net=host \
	      densedepth:latest \
              /bin/bash
xhost -local:root