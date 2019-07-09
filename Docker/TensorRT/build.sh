#!/bin/bash

docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg USERNAME=${USER} -t tensorrt:latest .
