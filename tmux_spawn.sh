#!/bin/bash

# user beware, completely untested.

echo "kicking off all scripts"
tmux new -d -s 0 'ros2 launch mavros px4.launch fcu_url:=/dev/ttyACM0:921600'
echo "waiting for mavros to boot"
sleep 5
tmux new -d -s 1 'cs ~/ws && colcon build --merge-install && source install/setup.sh && ros2 run libuavf_2024 esc_read.py'
tmux new -d -s 2 'cd ~/ws && colcon build --merge-install && source install/setup.sh && ros2 launch src/libuavf_2024/launches/imaging_demo.launch'
tmux new -d -s 3
tmux send-keys -t 3 "cs ~/ws && colcon build --merge-install && source install/setup.sh && ros2 run libuavf_2024 demo_commander_node.py $@" C-m
tmux a