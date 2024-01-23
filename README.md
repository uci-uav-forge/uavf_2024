# UAV Forge's ROS2 package for GN&C and Aerial Imagery Object Detection.

## Dev Container Setup (one time setup)
1. Ensure you have Docker and X-11 forwarding installed on your device. (google this)
2. Clone the repo and edit devcontainer.json accordingly for your X-11 setup. If you're on an non-linux OS you can also just comment out the lines for mounting the X socket and uncomment the lines for the "desktop-lite" below. Then by going to `localhost:6080` and entering the password `vscode` you should be able to access a desktop environment through your browser if necessary. 
3. In VSCode, first run `python3 setup_image.py` (alternatively run `echo "FROM t0mmyn/uavf_2024" > .devcontainer/Dockerfile` to download from a mirrored x86 image)
4. Then open the command palette (cmd/ctrl+shift+p) and run `rebuild and reopen in dev container`
5. To verify your setup, run `run_tests.sh`

## Dev Container Usage
1. Open VScode to the repo
2. Open the command palette (cmd/ctrl+shift+p) and run `rebuild and reopen in dev container`

## Sim Usage

1. Refer to `sim_instructions.md` for instructions on starting and running the simulation.


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
	1. Follow instructions in `sim_instructions.md`.
	2. If you want it to run it in a GUI, one way is using the remote desktop environment in the dev container. Open `localhost:6080` in a web browser, then enter password `vscode`, then use the menu in the bottom left to open a terminal.
	3. The X sockets should also be mounted and should work if you run `xhost +` on your machine.


I copied a lot of the config from this tutorial: https://docs.ros.org/en/foxy/How-To-Guides/Setup-ROS-2-with-VSCode-and-Docker-Container.html


### Manual

(WIP). It's recommended to use the Dockerfile for development.

### Running Imaging ROS Node
1. go into `/home/ws`
2. `source install/setup.sh`
3. `colcon build --merge-install`
4. `ros2 run uavf_2024 imaging_node.py` (this starts the service)
5. from another terminal, run `ros2 service call /imaging_service uavf_2024/srv/TakePicture`

### Jetson Installation
pytorch: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
as of jan 23, the jetson is on Jetpack 6
