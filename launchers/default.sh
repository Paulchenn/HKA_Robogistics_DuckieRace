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
dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/control_lane_node.py"


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
