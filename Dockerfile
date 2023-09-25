FROM ros:humble

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

VOLUME ["/root/ros2_ws/src/uavf_2024", "/root/ros2_ws/src/uavf_ros2_msgs"]

RUN apt-get update

# opencv dependencies (https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo)
RUN apt-get install -y ffmpeg libsm6 libxext6

RUN apt-get install -y python3-pip && \
    apt-get install -y ros-humble-mavros ros-humble-mavros-extras
RUN pip3 install --user -U empy pyros-genmsg setuptools

# utils (not strictly necessary)
RUN apt-get install -y tmux vim

# comment this out of you have a GPU
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY install_geographiclib_datasets.sh install_geographiclib_datasets.sh
RUN ./install_geographiclib_datasets.sh

COPY image_setup.sh image_setup.sh
RUN ./image_setup.sh

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /root/ros2_ws

CMD ["bash"]