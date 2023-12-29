# Instructions for testing using SITL

## Launch SITL

## Simple SITL

Run the following to start a simple SITL

```
ros2 launch ardupilot_sitl sitl_dds_udp.launch.py transport:=udp4 refs:=$(ros2 pkg prefix ardupilot_sitl)/share/ardupilot_sitl/config/dds_xrce_profile.xml synthetic_clock:=True wipe:=False model:=quad speedup:=1 slave:=0 instance:=0 defaults:=$(ros2 pkg prefix ardupilot_sitl)/share/ardupilot_sitl/config/default_params/copter.parm,$(ros2 pkg prefix ardupilot_sitl)/share/ardupilot_sitl/config/default_params/dds_udp.parm,/home/ws/uavf_2024/config/sitl.parm sim_address:=127.0.0.1 master:=tcp:127.0.0.1:5760 sitl:=127.0.0.1:5501 home:=38.31633,-76.55578,142,0
```

After starting the SITL, launch MAVProxy to visualize the drone position and send commands.

```
mavproxy.py --console --map --aircraft test --master=:14550
```

Follow https://ardupilot.org/dev/docs/copter-sitl-mavproxy-tutorial.html for instructions on using MAVProxy to control the sim.


## Start offboard control.

Launch MAVROS. (It converts ROS messages sent to it into commands sent to the flight control software.)

```
ros2 launch mavros apm.launch fcu_url:=udp://:14551@
```

Build uavf_2024.

```
cd /home/ws && colcon build --merge-install && source install/setup.bash
```

In the GCS (the window where you have `mavproxy` open), manually make the drone takeoff:
```
mode guided
arm throttle # might need to wait a bit for the sim to "heat up"
takeoff 20
```

Launch the mock imaging node:
```
ros2 run uavf_2024 mock_imaging_node.py /home/ws/uavf_2024/uavf_2024/gnc/data/AIRDROP_BOUNDARY 12 9
```

Launch the demo commander node:
```
ros2 run uavf_2024 demo_commander_node.py /home/ws/uavf_2024/uavf_2024/gnc/data/TEST_MISSION /home/ws/uavf_2024/uavf_2024/gnc/data/AIRDROP_BOUNDARY /home/ws/uavf_2024/uavf_2024/gnc/data/PAYLOAD_LIST 12 9
```

This will execute one lap of the mission in SITL.



### Other options: Gazebo
```
ros2 launch ardupilot_gz_bringup iris_runway.launch.py
```
Gazebo is another option for running SITL - so far we haven't worked the setup for this out.