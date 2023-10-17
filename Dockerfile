FROM ardupilot/ardupilot-dev-ros

VOLUME ["/root/ros2_ws/src/uavf_2024", "/root/ros2_ws/src/uavf_ros2_msgs"]

RUN apt-get update

# opencv dependencies (https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo)
RUN apt-get install -y ffmpeg libsm6 libxext6

# utils (not strictly necessary)
RUN apt-get install -y tmux vim

# comment this out if you have a GPU
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt


RUN apt-get install -y default-jre socat ros-humble-geographic-msgs ros-dev-tools

WORKDIR /root/ros2_ws/src

RUN wget https://raw.githubusercontent.com/ArduPilot/ardupilot/master/Tools/ros2/ros2.repos
RUN vcs import --recursive < ros2.repos
RUN wget https://raw.githubusercontent.com/ArduPilot/ardupilot_gz/main/ros2_gz.repos
RUN vcs import --recursive < ros2_gz.repos
ARG GZ_VERSION=garden

RUN apt-get -y install lsb-release wget gnupg
RUN wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
RUN apt-get update
RUN apt-get -y install gz-garden


WORKDIR /root/ros2_ws
RUN apt update
RUN rosdep update
RUN rosdep install --rosdistro ${ROS_DISTRO} --from-paths src -i -r -y
RUN bash -c "source /opt/ros/humble/setup.bash && colcon build --cmake-args -DBUILD_TESTING=ON"

RUN echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
RUN echo 'source /root/ros2_ws/install/setup.bash' >> ~/.bashrc

RUN echo 'export PATH=/root/ros2_ws/src/ardupilot/Tools/autotest:$PATH' >> ~/.bashrc
RUN echo 'export PATH=/usr/lib/ccache:$PATH' >> ~/.bashrc

RUN sudo apt-get install -y python3-dev python3-opencv python3-wxgtk4.0 python3-pip python3-matplotlib python3-lxml python3-pygame
RUN pip3 install PyYAML mavproxy --user
RUN echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc

RUN cp src/ardupilot/Tools/vagrant/mavinit.scr /root/.mavinit.scr
# i love python env management!!! <.<
RUN python -m pip uninstall matplotlib -y

CMD ["bash"]
