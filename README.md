# Sample repository showcasing model predictive control with PX4's offboard mode and ROS 2.

## Install instructions

### [ROS 2 Foxy for Ubuntu 20.04 Focal](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)

1. Uninstall ROS Noetic.
	```
	sudo apt-get remove ros-*
	sudo apt-get autoremove
	```
	Remove the installations of Noetic in /opt/ros and /mnt/c/opt/ros.
	```
	cd /opt/ros
	sudo rm -r noetic
	cd /mnt/c/opt/ros
	sudo rm -r noetic
	```
	Edit your ~/.bashrc and remove all ROS related commands.



2. Make sure locale has UTF-8.
	```
	locale  # check for UTF-8
	sudo apt update && sudo apt install locales
	sudo locale-gen en_US en_US.UTF-8
	sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
	export LANG=en_US.UTF-8
	locale  # verify settings
	```

3. Set up sources.
	```
	sudo apt install software-properties-common
	sudo add-apt-repository universe
	sudo apt update && sudo apt install curl -y
	sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
	echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
	```

4. Install ROS 2 packages.
	```
	sudo apt update
	sudo apt upgrade
	sudo apt install ros-foxy-desktop python3-argcomplete # full install with extra visualizer tools and tutorials (RECOMMENDED)
	sudo apt install ros-foxy-ros-base python3-argcomplete # only has communication libraries and command line tools
	sudo apt install ros-dev-tools
	```

5. Source environment and test if it's working.
	```
	source /opt/ros/foxy/setup.bash
	echo 'source /opt/ros/foxy/setup.bash' >> ~/.bashrc
	ros2 run demo_nodes_cpp talker
	```
	In another window run:
	```
	ros2 run demo_nodes_py listener
	```

6. Not sure what this is but PX4 says to install it.
	```
	pip install --user -U empy pyros-genmsg setuptools
	```


### [PX4 SITL for Ubuntu 20.04 Focal](https://docs.px4.io/main/en/ros/ros2_comm.html#foxy)

1. Install PX4 development environment in your directory of choice.
	```
	git clone https://github.com/PX4/PX4-Autopilot.git --recursive
	bash ./PX4-Autopilot/Tools/setup/ubuntu.sh
	cd PX4-Autopilot/
	make px4_sitl
	```


### [Micro XRCE-DDS Agent and Client (ROS 2 Middleware)](https://docs.px4.io/main/en/ros/ros2_comm.html#foxy)

1. Clone the DDS agent library to your directory of choice.
	```
	git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
	cd Micro-XRCE-DDS-Agent
	mkdir build
	cd build
	cmake ..
	make
	sudo make install
	sudo ldconfig /usr/local/lib/
	```
	If you get an error relating to "ASIO_INCLUDE_DIR" when running "make", install this package:
	```
	sudo apt-get install libasio-dev
	```

2. Start the DDS agent.
	```
	MicroXRCEAgent udp4 -p 8888
	```

3. The PX4 SITL contains the client. Run the SITL in a new client.
	```
	sudo make px4_sitl gazebo-classic
	```
	If you get a build error, check if you have any existing installations of ROS Noetic and delete them.
	After it successfully builds, you should see the simulation environment in one window and the DDS agent outputting "INFO" messages in its terminal window.


### [Set up PX4 ROS 2 workspace](https://docs.px4.io/main/en/ros/ros2_comm.html#foxy)

1. Create a ROS 2 workspace and name it whatever you want. This is where all you develop in.
	```
	mkdir -p ros2_ws/src
	```

2. Clone PX4's packages into your ROS 2 workspace and compile it.
	```
	cd ros2_ws/src
	git clone https://github.com/PX4/px4_msgs.git
	git clone https://github.com/PX4/px4_ros_com.git
	cd ..
	source /opt/ros/foxy/setup.bash
	colcon build

3. Run this command everytime you've compiled the workspace:
	```
	source install/local_setup.bash
	```

4. Test it out with an example
	```
	ros2 launch px4_ros_com sensor_combined_listener.launch.py
	```