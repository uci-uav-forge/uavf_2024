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
ros2 run libuavf_2024 mock_imaging_node.py /home/ws/libuavf_2024/uavf_2024/gnc/data/primary.gpx 12 9
```

Launch the demo commander node:
```
ros2 run libuavf_2024 demo_commander_node.py /home/ws/libuavf_2024/uavf_2024/gnc/data/primary.gpx /home/ws/libuavf_2024/uavf_2024/gnc/data/PAYLOAD_LIST 12 9 --is-maryland
```

This will execute one lap of the mission in SITL.

## To simulate at the ARC field:

### ARC club field sim:

Use the following commands:

To load a safety geofence like we will have in real life you may open /home/ws/libuavf_2024/uavf_2024/gnc/data/ARC/plan.plan in QGC.

```
cd /PX4-Autopilot
PX4_SIM_SPEED=2 PX4_HOME_LAT=33.64210 PX4_HOME_LON=-117.82683 PX4_HOME_ALT=142 make px4_sitl jmavsim
```

```
cd /home/ws && colcon build --merge-install && source install/setup.bash
ros2 run libuavf_2024 mock_imaging_node.py /home/ws/libuavf_2024/uavf_2024/gnc/data/club_field.gpx 12 9
```

```
cd /home/ws && colcon build --merge-install && source install/setup.bash
ros2 run libuavf_2024 demo_commander_node.py /home/ws/libuavf_2024/uavf_2024/gnc/data/club_field.gpx /home/ws/libuavf_2024/uavf_2024/gnc/data/PAYLOAD_LIST 12 9
```

ARC upper field sim:

```
cd /PX4-Autopilot
PX4_SIM_SPEED=2 PX4_HOME_LAT=33.64158 PX4_HOME_LON=-117.82573 PX4_HOME_ALT=142 make px4_sitl jmavsim
```

```
cd /home/ws && colcon build --merge-install && source install/setup.bash
ros2 run libuavf_2024 mock_imaging_node.py /home/ws/libuavf_2024/uavf_2024/gnc/data/upper_field.gpx 12 9
```

```
cd /home/ws && colcon build --merge-install && source install/setup.bash
ros2 run libuavf_2024 demo_commander_node.py /home/ws/libuavf_2024/uavf_2024/gnc/data/upper_field.gpx /home/ws/libuavf_2024/uavf_2024/gnc/data/PAYLOAD_LIST 12 9
```


## Steps to test mavlink radio messaging:
* Run `mavlink_console.py` to send statustext on GCS.

* `commander_node.py` now uses MAVROS to send statustext.

First visit, scanning dropzone.
bounds: [array([ 922.6684467271726 , -211.60795232067258]), array([1030.0019986414056 , -235.21765962940327]), array([1035.1171976997991, -213.3921503387189]), array([ 927.8943445436834 , -190.17227117260086]), array([ 922.6684467271726 , -211.60795232067258])]
homepos: mavros_msgs.msg.HomePosition(header=std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=1718316882, nanosec=339908520), frame_id=''), geo=geographic_msgs.msg.GeoPoint(latitude=38.3163299, longitude=-76.5557799, altitude=106.51997477569888), position=geometry_msgs.msg.Point(x=-0.008666543290019035, y=0.005678595043718815, z=-0.027722813189029694), orientation=geometry_msgs.msg.Quaternion(x=0.619987048410051, y=0.7846120591744359, z=3.7963257717409194e-17, w=4.8043632342019343e-17), approach=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=-0.0))
current xy: [1192.1728515625 -149.4306640625]
Closest corner is [1035.1171976997991 -213.3921503387189]
GPS: [ 38.31440644247138 -76.54394394390445]
Dropzone dimensions are 109.70826325955164x22.416915871830806
Dropzone unit vectors are [-0.22818478186918129 -0.973617843572782  ] and [-0.9773452789280257   0.21165114163901794]
Local coords: [(array([1029.350035253408, -218.28142726278 ]), 257.78086906802844), (array([1020.5539277430556 , -216.37656698802886]), 257.78086906802844), (array([1011.7578202327034 , -214.47170671327768]), 257.78086906802844), (array([1002.9617127223512, -212.5668464385265]), 257.78086906802844), (array([ 994.1656052119989 , -210.66198616377537]), 257.78086906802844), (array([ 985.3694977016467, -208.7571258890242]), 257.78086906802844), (array([ 976.5733901912945 , -206.85226561427305]), 257.78086906802844), (array([ 967.7772826809422 , -204.94740533952188]), 257.78086906802844), (array([ 958.98117517059  , -203.0425450647707]), 257.78086906802844), (array([ 950.1850676602378 , -201.13768479001956]), 257.78086906802844), (array([ 941.3889601498855, -199.2328245152684]), 257.78086906802844), (array([ 932.5928526395334 , -197.32796424051722]), 257.78086906802844), (array([ 923.7967451291811 , -195.42310396576607]), 257.78086906802844), (array([ 921.058527746751  , -207.10651808863946]), 193.19022578970765), (array([ 921.058527746751  , -207.10651808863946]), 77.78086906802844), (array([ 929.8546352571033, -209.0113783633906]), 77.78086906802844), (array([ 938.6507427674554 , -210.91623863814178]), 77.78086906802844), (array([ 947.4468502778077 , -212.82109891289295]), 77.78086906802844), (array([ 956.2429577881599, -214.7259591876441]), 77.78086906802844), (array([ 965.0390652985121 , -216.63081946239527]), 77.78086906802844), (array([ 973.8351728088644 , -218.53567973714644]), 77.78086906802844), (array([ 982.6312803192166 , -220.44054001189758]), 77.78086906802844), (array([ 991.4273878295688 , -222.34540028664875]), 77.78086906802844), (array([1000.2234953399211, -224.2502605613999]), 77.78086906802844), (array([1009.0196028502733 , -226.15512083615107]), 77.78086906802844), (array([1017.8157103606255 , -228.05998111090224]), 77.78086906802844), (array([1026.6118178709778, -229.9648413856534]), 77.78086906802844)]
Planned waypoints: [array([ 38.31436240253385, -76.5440098945023 ]), array([ 38.31437957310972, -76.54411046904197]), array([ 38.31439674359928, -76.54421104362906]), array([ 38.31441391400257, -76.5443116182636 ]), array([ 38.314431084319594, -76.54441219294557 ]), array([ 38.31444825455033, -76.54451276767497]), array([ 38.31446542469478, -76.54461334245181]), array([ 38.31448259475297, -76.54471391727608]), array([ 38.31449976472486, -76.54481449214778]), array([ 38.314516934610474, -76.54491506706691 ]), array([ 38.31453410440981, -76.54501564203348]), array([ 38.31455127412287, -76.54511621704748]), array([ 38.31456844374964, -76.5452167921089 ]), array([ 38.31446319289417, -76.54524811700769]), array([ 38.31446319289417, -76.54524811700769]), array([ 38.31444602329396, -76.54514754208425]), array([ 38.31442885360747, -76.54504696720824]), array([ 38.314411683834706, -76.54494639237966 ]), array([ 38.314394513975664, -76.54484581759851 ]), array([ 38.31437734403033, -76.5447452428648 ]), array([ 38.314360173998736, -76.54464466817852 ]), array([ 38.31434300388084, -76.54454409353967]), array([ 38.31432583367669, -76.54444351894826]), array([ 38.31430866338624, -76.54434294440428]), array([ 38.31429149300951, -76.54424236990774]), array([ 38.314274322546524, -76.54414179545861 ]), array([ 38.31425715199725, -76.54404122105693])]
Pushing waypoints
Waypoints: [(38.3149851, -76.5421146, 23.0), array([ 38.31436240253385, -76.5440098945023 ,  23.              ]), array([ 38.31436240253385, -76.5440098945023 ,  23.              ]), array([ 38.31437957310972, -76.54411046904197,  23.              ]), array([ 38.31439674359928, -76.54421104362906,  23.              ]), array([ 38.31441391400257, -76.5443116182636 ,  23.              ]), array([ 38.314431084319594, -76.54441219294557 ,  23.               ]), array([ 38.31444825455033, -76.54451276767497,  23.              ]), array([ 38.31446542469478, -76.54461334245181,  23.              ]), array([ 38.31448259475297, -76.54471391727608,  23.              ]), array([ 38.31449976472486, -76.54481449214778,  23.              ]), array([ 38.314516934610474, -76.54491506706691 ,  23.               ]), array([ 38.31453410440981, -76.54501564203348,  23.              ]), array([ 38.31455127412287, -76.54511621704748,  23.              ]), array([ 38.31456844374964, -76.5452167921089 ,  23.              ]), array([ 38.31446319289417, -76.54524811700769,  23.              ]), array([ 38.31446319289417, -76.54524811700769,  23.              ]), array([ 38.31444602329396, -76.54514754208425,  23.              ]), array([ 38.31442885360747, -76.54504696720824,  23.              ]), array([ 38.314411683834706, -76.54494639237966 ,  23.               ]), array([ 38.314394513975664, -76.54484581759851 ,  23.               ]), array([ 38.31437734403033, -76.5447452428648 ,  23.              ]), array([ 38.314360173998736, -76.54464466817852 ,  23.               ]), array([ 38.31434300388084, -76.54454409353967,  23.              ]), array([ 38.31432583367669, -76.54444351894826,  23.              ]), array([ 38.31430866338624, -76.54434294440428,  23.              ]), array([ 38.31429149300951, -76.54424236990774,  23.              ]), array([ 38.314274322546524, -76.54414179545861 ,  23.               ]), array([ 38.31425715199725, -76.54404122105693,  23.              ])] Yaws: [nan, nan, 257.78086906802844, 257.78086906802844, 257.78086906802844, 257.78086906802844, 257.78086906802844, 257.78086906802844, 257.78086906802844, 257.78086906802844, 257.78086906802844, 257.78086906802844, 257.78086906802844, 257.78086906802844, 257.78086906802844, 193.19022578970765, 77.78086906802844, 77.78086906802844, 77.78086906802844, 77.78086906802844, 77.78086906802844, 77.78086906802844, 77.78086906802844, 77.78086906802844, 77.78086906802844, 77.78086906802844, 77.78086906802844, 77.78086906802844, 77.78086906802844]
Delaying before pushing waypoints.
Pushing waypoints.
Delaying before resetting mission progress.
Set mission progress
Delaying before setting mode.