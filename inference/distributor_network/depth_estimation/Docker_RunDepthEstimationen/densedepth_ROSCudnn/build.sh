#!/bin/bash

docker build --no-cache --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg USERNAME=${USER} -t densedepth_roscudnn:latest .
