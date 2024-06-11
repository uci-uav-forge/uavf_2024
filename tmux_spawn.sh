#!/bin/bash

# user beware, completely untested.

echo "kicking off all scripts"
tmux new -d
tmux new-window 'ros2 launch mavros px4.launch fcu_url:=/dev/ttyACM0:921600'
echo "waiting for mavros to boot"
sleep 5
tmux new-window 'cd ~/ws && colcon build --merge-install && source install/setup.sh && ros2 run libuavf_2024 esc_read.py;bash'
tmux new-window 'cd ~/ws && colcon build --merge-install && source install/setup.sh && ros2 launch src/libuavf_2024/launches/imaging_demo.launch;bash'
tmux new-window
tmux send-keys -t 0:4 "cd ~/ws && colcon build --merge-install && source install/setup.sh && ros2 run libuavf_2024 demo_commander_node.py $@" C-m
tmux a
