FROM ros:humble

RUN apt update
RUN apt-get update --fix-missing
RUN apt-get install -y python3-pip
# RUN apt install -y curl
# RUN apt-get install -y wget

# RUN git clone --recursive https://github.com/PX4/PX4-Autopilot.git /home/ws/PX4-Autopilot
# RUN curl -sSL http://get.gazebosim.org | sh

# WORKDIR "/home/ws/PX4-Autopilot"
# RUN Tools/setup/ubuntu.sh
# RUN make px4_sitl

CMD ["/bin/bash"]