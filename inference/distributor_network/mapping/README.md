# Mapping

## Requirements

The mapping tool should work on any linux PC with nvidia graphics card and at least cuda 10.0. Changes might be necessary to run the tool on a PC without nvidia graphics card.

Nvidia-Docker is required to build and run the docker containers.

## Setup

After cloning the repository should download the weights for the indoor segmentation as described on https://github.com/hellochick/Indoor-segmentation and put the folder ResNet101 into ./home/Mapping/restore_weights/

It is also necessary to build to build the two docker containers. For that purpose execute the build.sh files in ./Docker_ORB2_SLAM/ and in ./Docker_Python/

After that you should use the launch.sh in ./Docker_ORB2_SLAM/ to launch the container and build ORB2 with ./ORB_SLAM2/build.sh

## Run the Mapping

Use the launch.sh files in ./Docker_ORB2_SLAM/ and in ./Docker_Python/ to start the two containers. In the python container use 

cd ./Mapping
python3 main.py --input ~/host/path_to_video 

to launch the mapper

In the ORB2 container use 

cd ./ORB_SLAM2/Examples/Monocular
./mono /../../Vocabulary/ORBvoc.txt /../../config/appropriate_config_file.yaml

to launch ORB2.

The mapping results are saved to ./home/Mapping/maps

The ./config folder contains example .yaml files for an number of camera configurations
