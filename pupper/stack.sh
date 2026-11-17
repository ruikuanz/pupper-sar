#!/bin/bash
# The robot stack (pupper/launch/pupper.launch.py): start it, stop it.
#   pupper/stack.sh start [log-dir] [--assistant]
#                         start it if it is not running; wait until the walking controller is loaded.
#                         --assistant also loads the trick controllers and the person follower.
#   pupper/stack.sh stop  stop it and every ROS process it left behind
#   pupper/stack.sh busy <me>   exit 1 (and say which) if another of this project's servers is running
# surfer/, echo/ and gemini/start.sh call this; you rarely need to.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=/tmp/pupper-sar            # walk_controller.py's RUN_DIR
WS=${PUPPER_ROS_WS:-$HOME/pupperv3-monorepo/ros2_ws}
pattern() { echo "[${1:0:1}]${1:1}"; }   # "[/]opt/ros": a pattern that does not match the pkill looking for it

case "${1:-}" in
busy)
  # Only one server may drive the legs at a time.
  for s in surfer/server.py echo/server.py gemini/server.py; do
    [ "$s" = "${2:-}" ] && continue
    if pgrep -f "$(pattern "$s")" >/dev/null; then
      echo "$s is running and drives the legs; stop it first (${s%%/*}/stop.sh)"; exit 1
    fi
  done
  ;;
start)
  LOG=${2:-$RUN}; mkdir -p "$LOG" "$RUN"
  [ "${3:-}" = "--assistant" ] && export SAR_ASSISTANT=1
  [ -f /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash ] || { echo "ROS 2 is not installed here: this runs on the Pupper (on a laptop, use --sim)"; exit 1; }
  [ -f "$WS/install/local_setup.bash" ] || { echo "No Pupper ROS workspace at $WS (set PUPPER_ROS_WS if it is somewhere else)"; exit 1; }
  if pgrep -f "[r]os2_control_node" >/dev/null; then
    if pgrep -f "[c]md_vel_mux_node" >/dev/null; then
      echo "      robot stack already running"
      if [ "${SAR_ASSISTANT:-0}" = 1 ] && [ "$(cat $RUN/mode 2>/dev/null)" != assistant ]; then
        echo "      (started without --assistant: tricks load on first use, a few seconds each, and follow mode"
        echo "       may be missing. For both: pupper/stack.sh stop, then start again.)"
      fi
    else
      # Someone else's stack: the motors are up, but nothing forwards our speed commands.
      echo "      a robot stack is running, but not ours (no cmd_vel_mux). Stop it first:  pupper/stack.sh stop"; exit 1
    fi
  else
    echo "      starting the robot stack (about a minute; log $LOG/robot.log)"
    rm -f $RUN/controllers_ready
    [ "${SAR_ASSISTANT:-0}" = 1 ] && echo assistant > $RUN/mode || echo plain > $RUN/mode
    [ "${SAR_ASSISTANT:-0}" = 1 ] && echo "      with the trick controllers (a few minutes on first start)"
    setsid nohup bash -c "source $HERE/ros_env.sh; exec ros2 launch $HERE/launch/pupper.launch.py" > "$LOG/robot.log" 2>&1 < /dev/null &
    # walk_controller.py writes this once sar_walk is loaded (or has failed to).
    for i in $(seq 1 240); do [ -f $RUN/controllers_ready ] && break; sleep 1; done
    if [ -f $RUN/controllers_ready ]; then echo "      $(cat $RUN/controllers_ready)"
    else echo "      the walking controller is still loading after 4 min; see $LOG/robot.log"; fi
  fi
  ;;
stop)
  pkill -INT -f "[r]os2 launch" 2>/dev/null
  for i in $(seq 1 15); do pgrep -f "[r]os2_control_node" >/dev/null || break; sleep 1; done
  # Whatever did not exit on its own.
  pkill -9 -f "[r]os2 launch" 2>/dev/null
  pkill -9 -f "$(pattern /opt/ros/${ROS_DISTRO:-jazzy}/lib/)" 2>/dev/null
  pkill -9 -f "$(pattern "$WS/install/")" 2>/dev/null
  rm -f $RUN/controllers_ready $RUN/mode
  ;;
*)
  echo "usage: pupper/stack.sh start [log-dir] [--assistant] | stop | busy <server>"; exit 2
  ;;
esac
