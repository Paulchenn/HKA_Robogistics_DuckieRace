#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch camera node
dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/camera_reader_node.py" &

# launch intersection handling node
dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/intersection_handling_node.py" &

# launch switch control node
dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/switch_control_node.py" &

# launch control lane node
dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/control_lane_node.py" &

# launch control lane node
dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/turning_process_node.py" 

# wait for app to end
dt-launchfile-join