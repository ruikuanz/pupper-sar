#!/bin/bash
# Start Pupper as a player. Run on the Pupper, from the repo:
#   surfer/start.sh                      eyes on the streamed game (start here)
#   surfer/start.sh --source camera      eyes on Pupper's own camera, looking at the screen
#   surfer/start.sh --source camera --hold --view-width 0.20
# Any argument is passed on to surfer/server.py (see its --help).
# Stop everything with surfer/stop.sh.
#
# Step 1 is the robot stack (pupper/stack.sh): motors, IMU, camera, the walking
# policy, and the joystick outranking us. Step 2 is the player.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE")
LOG=/tmp/lane-hound; mkdir -p $LOG
PORT=8770

[ -f "$HERE/references/squat.json" ] || python3 "$HERE/squat_reference.py"

"$ROOT/pupper/stack.sh" busy surfer/server.py || exit 1
echo "[1/2] the robot stack"
"$ROOT/pupper/stack.sh" start $LOG || exit 1

echo "[2/2] starting the player on localhost:$PORT (log $LOG/server.log)"
pkill -f "[s]urfer/server.py" 2>/dev/null; sleep 1
setsid nohup bash -c "source $ROOT/pupper/ros_env.sh; exec python3 -u $HERE/server.py $*" > $LOG/server.log 2>&1 < /dev/null &
for i in $(seq 1 30); do curl -s -o /dev/null localhost:$PORT/status && break; sleep 1; done
curl -s -o /dev/null localhost:$PORT/status || { echo "the player did not start; see $LOG/server.log"; tail -5 $LOG/server.log; exit 1; }
echo "ready. From your laptop:  ssh -N -L $PORT:localhost:$PORT pi@<pupper>.local   then open http://localhost:$PORT"
