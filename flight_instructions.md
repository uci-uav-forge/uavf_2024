# Flight instructions

Boot up PX4 on the ThinkPad and make sure you have a good connection to the drone.

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

```
cd ~/ws && colcon build --merge-install && source install/setup.bash
ros2 run libuavf_2024 mock_imaging_node.py ~/ws/src/libuavf_2024/uavf_2024/gnc/data/ARC/AIRDROP_BOUNDARY 12 9
```

Launch the commander node (for now, note the `end-early` flag - this terminates the script after the mission is pushed and completed.) 

```
cd ~/ws && colcon build --merge-install && source install/setup.bash
ros2 run libuavf_2024 demo_commander_node.py ~/ws/src/libuavf_2024/uavf_2024/gnc/data/ARC/UPPER_FIELD_MISSION ~/ws/src/libuavf_2024/uavf_2024/gnc/data/ARC/AIRDROP_BOUNDARY ~/ws/src/libuavf_2024/uavf_2024/gnc/data/PAYLOAD_LIST 12 9 --exit-early
```
