# Instructions for running the Gazebo sim

## Set up params
```
sim_vehicle.py -w -v ArduCopter --console -DG --enable-dds
```
This command builds the SITL for arducopter and wipes existing params.

Once it's started, type the following command and then force-quit:
```
param set DDS_ENABLE 1
```

## Launch SITL

## Starting the sim

After this you have a few different options. 

### Launch sim_vehicle (recommended)
```
sim_vehicle.py --map --console -v ArduCopter
```
Follow https://ardupilot.org/dev/docs/copter-sitl-mavproxy-tutorial.html for instructions on using MAVProxy to control the sim.

### Launch Gazebo
```
# In one terminal...
ros2 launch ardupilot_sitl sitl_dds_udp.launch.py transport:=udp4 refs:=$(ros2 pkg prefix ardupilot_sitl)/share/ardupilot_sitl/config/dds_xrce_profile.xml synthetic_clock:=True wipe:=False model:=quad speedup:=1 slave:=0 instance:=0 defaults:=$(ros2 pkg prefix ardupilot_sitl)/share/ardupilot_sitl/config/default_params/copter.parm,$(ros2 pkg prefix ardupilot_sitl)/share/ardupilot_sitl/config/default_params/dds_udp.parm sim_address:=127.0.0.1 master:=tcp:127.0.0.1:5760 sitl:=127.0.0.1:5501
# In a separate terminal
ros2 launch ardupilot_gz_bringup iris_runway.launch.py
# In a third terminal launch MAVProxy or another GCS
```
Gazebo is another option, but I found it very slow through the layer of virtualization on my machine... 


