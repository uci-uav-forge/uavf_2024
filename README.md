# UAV Forge's ROS2 package for GN&C and Aerial Imagery Object Detection.

## Dev Container Setup
1. Clone the repo
2. In VSCode, open the command palette and run `rebuild and reopen in dev container`
3. To verify your setup, run `run_tests.sh`

## Usage

1. Start the DDS agent.
	```
	MicroXRCEAgent udp4 -p 8888
	```

2. In a new terminal window, cd into PX4-Autopilot and start the SITL.
	```
	sudo make px4_sitl gazebo-classic
	```

3. In a new terminal window, run your roslaunch or script.


## Install instructions

### Install required and local Python libraries

1. cd into this repo's root directory.

2. Run:
	```
	pip install -e .
	```


### Dev container

1. Open this project in vscode
2. Install the "Dev Containers" extension
3. Open the command pallete (ctrl-shift-p), then search for and execute "Dev Containers: (Re-)build and Reopen in Container"
4. Congratulations, you get to skip all those tedious steps to install ROS 2 manually, and your environment is isolated from the rest of your computer
5. To make downloading dependencies reproducible, add any important software installation steps to the Dockerfile in this repo.
6. To use git inside the docker container, you may have to manually log in to GitHub again if the built-in credential forwarding isn't working. I recommend using the [GitHub CLI](https://cli.github.com/) to do this.
7. If you want to use the simulator:
	1. If you want to run in headless, `cd /home/ws/PX4-Autopilot` then `HEADLESS=1 make px4_sitl gazebo-classic`
	2. If you want it to run it in a GUI, one way is using the remote desktop environment in the dev container. Open `localhost:6080` in a web browser, then enter password `vscode`, then use the menu in the bottom left to open a terminal, `cd /home/ws/PX4-Autopilot`, then run `make px4_sitl gazebo-classic`.


I copied a lot of the config from this tutorial: https://docs.ros.org/en/foxy/How-To-Guides/Setup-ROS-2-with-VSCode-and-Docker-Container.html


### Manual

#### [ROS 2 Foxy for Ubuntu 20.04 Focal](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)

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

3. The PX4 SITL contains the DDS client. Run the SITL in a new window.
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

2. Clone PX4's packages and [our custom messages](https://github.com/Herpderk/uavf_msgs/tree/master/msg) into your ROS 2 workspace and compile it.
	```
	cd ros2_ws/src
	git clone https://github.com/PX4/px4_msgs.git
	git clone git@github.com:uci-uav-forge/uavf_ros2_msgs.git
 	git clone git@github.com:uci-uav-forge/uavf_2024.git
	cd ..
	source /opt/ros/foxy/setup.bash
	```
	Everytime you make a change to your workspace, rebuild with this command:
	```
	colcon build --merge-install
	```
	If you compile with only "colcon build", you will get import errors with your ROS message types.
	

3. Run this command after everytime you've built the workspace:
	```
	source install/setup.bash
	```

4. Test it out with an example
	```
	ros2 run uavf_2024 commander_node.py
	```


### Acados Installation (for MPC usage)

1. [Follow these instructions to install Acados.](https://docs.acados.org/installation/)

2. [Install the Python interface afterwards.](https://docs.acados.org/python_interface/index.html)
