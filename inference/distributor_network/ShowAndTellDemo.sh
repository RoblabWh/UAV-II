#!/bin/sh

echo Starting segmentation...
python3 ./segmentation/voc_new.py --host_ip '192.168.178.41' --port 5555 &

echo Starting pose-detection...
python3 ./pose_detection/pose_detection.py --host_ip '192.168.178.41' --port 5555 &
