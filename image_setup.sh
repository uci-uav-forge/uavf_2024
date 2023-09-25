#!/bin/bash

git clone https://github.com/PX4/PX4-Autopilot.git --recursive
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh
cd PX4-Autopilot/
make px4_sitl
cd /

git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build
cd build
cmake ..
make
make install
ldconfig /usr/local/lib/
cd /

mkdir -p /root/src/
cd /root/src/
git clone https://github.com/PX4/px4_msgs.git
source /opt/ros/humble/setup.bash
colcon build --merge-install # takes like 7 minutes to build...
