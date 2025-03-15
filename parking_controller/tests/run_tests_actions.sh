#!/bin/bash -l

source /etc/bash.bashrc

set -eux

id
ls /

cd $HOME/racecar_ws

colcon build --symlink-install
set +eux
source ./install/setup.bash
set -eux

cd ./src/parking_controller
direnv allow
eval "$(direnv export bash)"
sudo pytest
