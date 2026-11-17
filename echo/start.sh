#!/bin/bash
# Start Pupper finding its way by ear. Run on the Pupper, from the repo:
#   echo/start.sh                 two ears: the compact USB-C receiver, two mics (start here)
#   echo/start.sh --ears four     the main receiver, up to four mics
#   echo/start.sh --ears two --mic "Wireless" --speaker 3
# Any argument is passed on to echo/server.py (see its --help).
# Stop everything with echo/stop.sh.
#
# Run echo/mic_check.py first: this script assumes the ears already work.
# Step 1 is the robot stack (pupper/stack.sh): motors, IMU, the walking policy,
# and the joystick outranking us. Step 2 is the ears.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE")
LOG=/tmp/pupper-echo; mkdir -p $LOG
PORT=8771

python3 -c "import sounddevice" 2>/dev/null || { echo "sounddevice is missing:  sudo apt install libportaudio2 && pip install --user --break-system-packages sounddevice"; exit 1; }

"$ROOT/pupper/stack.sh" busy echo/server.py || exit 1
echo "[1/2] the robot stack"
"$ROOT/pupper/stack.sh" start $LOG || exit 1

# The chirp has to be heard over the room: speaker all the way up.
amixer -q sset Master 100% 2>/dev/null; wpctl set-volume @DEFAULT_SINK@ 1.0 2>/dev/null

echo "[2/2] starting the ears on localhost:$PORT (log $LOG/server.log)"
pkill -f "[e]cho/server.py" 2>/dev/null; sleep 1
setsid nohup bash -c "source $ROOT/pupper/ros_env.sh; exec python3 -u $HERE/server.py $*" > $LOG/server.log 2>&1 < /dev/null &
for i in $(seq 1 30); do curl -s -o /dev/null localhost:$PORT/status && break; sleep 1; done
curl -s -o /dev/null localhost:$PORT/status || { echo "the server did not start; see $LOG/server.log"; tail -5 $LOG/server.log; exit 1; }
echo "ready. From your laptop:  ssh -N -L $PORT:localhost:$PORT pi@<pupper>.local   then open http://localhost:$PORT"
