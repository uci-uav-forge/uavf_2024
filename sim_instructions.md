# Instructions for testing using SITL

## Launch SITL

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

