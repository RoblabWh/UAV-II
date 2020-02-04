#!/bin/sh

echo Starting Segmentation...
terminator -T "Segmentation" -e "python3 ./segmentation/voc_new.py --cmap './segmentation/cmap.npy' --host_ip '172.16.35.164' --port 5555" &

echo Starting PoseDetection...
terminator -T "PoseDetection" -e "python3 ./pose_detection/pose_detection.py --host_ip '172.16.35.164' --port 5555" &
