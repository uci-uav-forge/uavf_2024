# Flight instructions

Boot up QGC on the ThinkPad and make sure you have a good connection to the drone.

Start MAVROS:

```
ros2 launch mavros px4.launch fcu_url:=/dev/ttyACM0:921600
```

If this isn't working with an error message like `serial open permission denied`, you need to run this setup step: `sudo chmod 666 /dev/ttyACM0`

At this point it would be a good idea to make sure all is good with MAVROS topics; poke around a bit with

```
ros2 topic list
ros2 topic echo /mavros/state # etc
```


Launch the demo (for now) imaging node.

(ARC Upper field)

```
cd ~/ws && colcon build --merge-install && source install/setup.bash
ros2 run libuavf_2024 mock_imaging_node.py ~/ws/src/libuavf_2024/uavf_2024/gnc/data/ARC/UPPER_FIELD_DROPZONE 12 9
```

(ARC Club field)

```
cd ~/ws && colcon build --merge-install && source install/setup.bash
ros2 run libuavf_2024 mock_imaging_node.py ~/ws/src/libuavf_2024/uavf_2024/gnc/data/ARC/CLUB_FIELD/AIRDROP_BOUNDARY 12 9
```

Launch the commander node.
There are a few relevant flags that one might want to use for testing
* `--exit-early`: Exits after pushing first waypoint mission
* `--servo-test`: Don't do anything mission-related, just actuate the servo and quit.
* `--call-imaging` (and its optional partner `--call-imaging-period`): Don't do anything mission related, just call the imaging service, convert the coord to GPS, and print.

(ARC Upper field)

```
cd ~/ws && colcon build --merge-install && source install/setup.bash
ros2 run libuavf_2024 demo_commander_node.py ~/ws/src/libuavf_2024/uavf_2024/gnc/data/upper_field.gpx ~/ws/src/libuavf_2024/uavf_2024/gnc/data/PAYLOAD_LIST 12 9
```

(ARC Club field)

```
cd ~/ws && colcon build --merge-install && source install/setup.bash
ros2 run libuavf_2024 demo_commander_node.py ~/ws/src/libuavf_2024/uavf_2024/gnc/data/ARC/club_field.gpx ~/ws/src/libuavf_2024/uavf_2024/gnc/data/PAYLOAD_LIST 12 9
```
