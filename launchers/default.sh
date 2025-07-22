#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# launching app
# dt-exec echo "This is an empty launch script. Update it to launch your application."

# launching camera_reader_node
# dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/camera_reader_node.py"

# launching control_lane_node
#ls -l "$(dirname "$0")"

# 1. Steuerung aktivieren
dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/switch_control_node.py" &

# 2. Lane Detection starten
#dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/detect_lane_node.py" &

dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/camera_reader_node.py" &

dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/control_lane_node.py" &

dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/detect_object_node.py" &

dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/bypass_duckie_node.py" &

dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/display_yoloResult_node.py"

# dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/intersection_handling_node.py"

#dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/parking_node.py" 

# dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/collision_avoidance_node.py" 

#dt-exec python3 "$DT_REPO_PATH/packages/testtim/my_script.py" 

# 3. Spur folgen
#dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/control_lane_node.py"

#dt-exec bash /launch/HKA_Robogistics_DuckieRace/camera-reader.sh &


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
