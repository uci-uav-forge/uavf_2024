# Instructions for testing using SITL

You might need to run the following on the host machine before building the Docker image:
```
git submodule init siyi_sdk && git submodule update siyi_sdk
```

## Simple SITL

Run the following to start a simple SITL

```
cd /PX4-Autopilot
PX4_SIM_SPEED=2 PX4_HOME_LAT=38.31633 PX4_HOME_LON=-76.55578 PX4_HOME_ALT=142 make px4_sitl jmavsim
```

After starting the SITL, launch QGC.

```
sudo -H -u qgc /QGroundControl.AppImage
```

Ask QGC to takeoff using the UI.

## Headless remote SITL (no Docker needed!)

Follow these instructions by Eric to establish a connection & SSH to the lab machine.

https://docs.google.com/document/d/1yx_y53GlGXzIlb5XoCMI9oupPpIu1cXJN4InP-qLm7g/edit#heading=h.8zsd8msotkte

Then run this command to start the docker container `docker run -it -v ~/uavf_2024:/home/ws/libuavf_2024 3940da8882a0`. The image ID (last argument) might be different, but you can find it by just running `docker image ls` and looking for the big ones from VS code.

Once you're in the container, run `sudo apt install tmux` and then run `tmux`.


To start jmavsim in headless mode, prepend `HEADLESS=1`.
```
cd /PX4-Autopilot
HEADLESS=1 PX4_SIM_SPEED=2 PX4_HOME_LAT=38.31633 PX4_HOME_LON=-76.55578 PX4_HOME_ALT=142 make px4_sitl jmavsim
```

You can skip QGroundcontrol and just type the following in the `jmavsim` window to takeoff.
```
commander takeoff
```

After this you can go on to follow the rest of the instructions as normal.


## Start offboard control.

Launch MAVROS. (It converts ROS messages sent to it into commands sent to the flight control software.)

```
ros2 launch mavros px4.launch fcu_url:=udp://:14540@
```

Build uavf_2024.

```
cd /home/ws && colcon build --merge-install && source install/setup.bash
```

Launch the mock imaging node:
```
ros2 run libuavf_2024 mock_imaging_node.py /home/ws/libuavf_2024/uavf_2024/gnc/data/AIRDROP_BOUNDARY 12 9
```

Launch the demo commander node:
```
ros2 run libuavf_2024 demo_commander_node.py /home/ws/libuavf_2024/uavf_2024/gnc/data/TEST_MISSION /home/ws/libuavf_2024/uavf_2024/gnc/data/AIRDROP_BOUNDARY /home/ws/libuavf_2024/uavf_2024/gnc/data/PAYLOAD_LIST 12 9
```

This will execute one lap of the mission in SITL.

## To simulate at the ARC field:

### ARC club field sim:

Use the following commands:

To load a safety geofence like we will have in real life you may open /home/ws/libuavf_2024/uavf_2024/gnc/data/ARC/plan.plan in QGC.

```
cd /PX4-Autopilot
PX4_SIM_SPEED=2 PX4_HOME_LAT=33.64230 PX4_HOME_LON=-117.82683 PX4_HOME_ALT=142 make px4_sitl jmavsim
```

```
ros2 run libuavf_2024 mock_imaging_node.py /home/ws/libuavf_2024/uavf_2024/gnc/data/ARC/CLUB_FIELD/AIRDROP_BOUNDARY 12 9
```

```
ros2 run libuavf_2024 demo_commander_node.py /home/ws/libuavf_2024/uavf_2024/gnc/data/ARC/CLUB_FIELD/MISSION /home/ws/libuavf_2024/uavf_2024/gnc/data/ARC/CLUB_FIELD/AIRDROP_BOUNDARY /home/ws/libuavf_2024/uavf_2024/gnc/data/PAYLOAD_LIST 12 9 --exit-early
```

ARC upper field sim:

```
cd /PX4-Autopilot
PX4_SIM_SPEED=2 PX4_HOME_LAT=33.64158 PX4_HOME_LON=-117.82573 PX4_HOME_ALT=142 make px4_sitl jmavsim
```

```
ros2 run libuavf_2024 mock_imaging_node.py /home/ws/libuavf_2024/uavf_2024/gnc/data/ARC/UPPER_FIELD_DROPZONE 12 9
```

```
ros2 run libuavf_2024 demo_commander_node.py /home/ws/libuavf_2024/uavf_2024/gnc/data/ARC/UPPER_FIELD_MISSION /home/ws/libuavf_2024/uavf_2024/gnc/data/ARC/UPPER_FIELD_DROPZONE /home/ws/libuavf_2024/uavf_2024/gnc/data/PAYLOAD_LIST 12 9
```
