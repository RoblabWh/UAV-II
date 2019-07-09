FROM densedepthpyqt5:latest
MAINTAINER roblabfhge <roblab2011@googlemail.com>
 
# Define build arguments
ARG USER_ID
ARG GROUP_ID
ARG USERNAME

USER $USERNAME
ENV HOME /home/${USERNAME}

RUN sudo usermod -a -G video ${USERNAME}
