FROM densedepth:latest
MAINTAINER roblabfhge <roblab2011@googlemail.com>

RUN sudo pip --no-cache-dir uninstall -y pySide2
RUN sudo apt-get update
RUN sudo pip --no-cache-dir install PyQt5
RUN sudo pip --no-cache-dir install numpy
RUN sudo pip --no-cache-dir install imageio
RUN sudo pip --no-cache-dir install scipy
 
# Define build arguments
ARG USER_ID
ARG GROUP_ID
ARG USERNAME

USER $USERNAME
ENV HOME /home/${USERNAME}

RUN sudo usermod -a -G video ${USERNAME}
