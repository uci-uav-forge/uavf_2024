# Instructions for running the Gazebo sim

## Set up params
```
sim_vehicle.py -w -v ArduCopter --console -DG --enable-dds
```
This command builds the SITL for arducopter and wipes existing params.

Once it's started, type the following command and then control-c:
```
param set DDS_ENABLE 1
```

## Launch SITL

## Simple SITL

Run the following to start a simple SITL

```
ros2 launch ardupilot_sitl sitl_dds_udp.launch.py transport:=udp4 refs:=$(ros2 pkg prefix ardupilot_sitl)/share/ardupilot_sitl/config/dds_xrce_profile.xml synthetic_clock:=True wipe:=False model:=quad speedup:=1 slave:=0 instance:=0 defaults:=$(ros2 pkg prefix ardupilot_sitl)/share/ardupilot_sitl/config/default_params/copter.parm,$(ros2 pkg prefix ardupilot_sitl)/share/ardupilot_sitl/config/default_params/dds_udp.parm sim_address:=127.0.0.1 master:=tcp:127.0.0.1:5760 sitl:=127.0.0.1:5501 home:=38.31633,-76.55578,142,0
```

After starting the SITL one options is connecting to it and sending commands directly with MAVProxy.

```
mavproxy.py --console --map --aircraft test --master=:14550
```

Follow https://ardupilot.org/dev/docs/copter-sitl-mavproxy-tutorial.html for instructions on using MAVProxy to control the sim.


### Launch Gazebo
```
ros2 launch ardupilot_gz_bringup iris_runway.launch.py
```
Gazebo is another option, but I found it very slow through the layer of virtualization on my machine, and harder to set up. 


## Start offboard control.

Launch MAVROS. (It converts ROS messages sent to it into commands sent to the flight control software.)

```
ros2 launch mavros apm.launch fcu_url:=udp://:14551@
```


