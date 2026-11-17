# The ROS environment on Pupper. Source it; do not run it.
#   ROS 2 itself, then the pupperv3 workspace the robot ships with (motor
#   drivers, the walking policy's controller, cmd_vel_mux, the camera).
# If your workspace is somewhere else:  export PUPPER_ROS_WS=/path/to/ros2_ws
source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash
source "${PUPPER_ROS_WS:-$HOME/pupperv3-monorepo/ros2_ws}/install/local_setup.bash"
export ROS_LOCALHOST_ONLY=1   # ROS traffic stays on the robot: other Puppers on the network do not hear it
