#!/bin/bash

# Sourcing things
echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc
echo 'export PATH=/usr/lib/ccache:$PATH' >> /root/.bashrc

echo 'export PATH=$PATH:$HOME/.local/bin' >> /root/.bashrc

# So that the container doesn't close
/bin/bash -i