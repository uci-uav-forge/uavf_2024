# set to arm64 version for macs with apple sillicon
FROM arm64v8/ros:humble

RUN apt update
RUN apt-get update --fix-missing
RUN apt-get install -y python3-pip
RUN apt-get install -y ffmpeg libsm6 libxext6
# RUN apt install -y curl
# RUN apt-get install -y wget

# RUN git clone --recursive https://github.com/PX4/PX4-Autopilot.git /home/ws/PX4-Autopilot
# RUN curl -sSL http://get.gazebosim.org | sh

# WORKDIR "/home/ws/PX4-Autopilot"
# RUN Tools/setup/ubuntu.sh
# RUN make px4_sitl
WORKDIR "/home/ws/uavf_2024"
COPY setup.py setup.py
RUN pip install -e .

# downgrading torch libararies to prevent segmentation faults
# torch==2.0.1 
# torchvision==0.15.2 
# torchaudio==2.0.2
RUN pip install --force-reinstall -v torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

CMD ["/bin/bash"]