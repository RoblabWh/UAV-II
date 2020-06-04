FROM densedepthcudnn:latest
MAINTAINER roblabfhge <roblab2011@googlemail.com>


RUN /bin/bash -c "sudo sh -c 'echo \"deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main\" > /etc/apt/sources.list.d/ros-latest.list' && \
    sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    sudo apt update && \
    sudo apt install -y ros-melodic-desktop-full"


RUN /bin/bash -c "sudo pip install -U rosdep"
RUN /bin/bash -c "sudo rosdep init"
RUN /bin/bash -c "rosdep update"

RUN /bin/bash -c "echo \"source /opt/ros/melodic/setup.bash\" >> ~/.bashrc && source ~/.bashrc"

RUN /bin/bash -c "sudo apt install -y python-rosinstall python-rosinstall-generator python-wstool build-essential"

RUN /bin/bash -c "source ~/.bashrc && mkdir -p ~/catkin_ws/src && cd ~/catkin_ws/ && catkin_make"

RUN /bin/bash -c "sudo apt-get install -y python3-pip python3-yaml && sudo pip3 install rospkg catkin_pkg"

RUN /bin/bash -c "sudo apt-get install -y python-catkin-tools python3-dev python3-numpy"

RUN /bin/bash -c "source ~/.bashrc && mkdir ~/catkin_build_ws && cd ~/catkin_build_ws && \
    catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so && \
    catkin config --install"

RUN /bin/bash -c "source ~/.bashrc && mkdir ~/catkin_build_ws/src && cd ~/catkin_build_ws/src && git clone -b melodic https://github.com/ros-perception/vision_opencv.git && cd ~/catkin_build_ws && catkin build cv_bridge && source install/setup.bash --extend"

RUN /bin/bash -c "sudo apt-get install -y ros-melodic-rtabmap-ros"

RUN sudo pip3 install tensorflow
RUN sudo pip3 install tensorflow-addons
