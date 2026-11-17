#!/bin/bash
# Start the assistant: talk to Pupper, and it listens, looks and does what you ask.
# Run on the Pupper, from the repo:
#   gemini/start.sh            robot stack + robot server (open the app from a laptop through an SSH tunnel)
#   gemini/start.sh --kiosk    also show the app full-screen on Pupper's own screen, using its mic and speaker
# Stop everything with gemini/stop.sh. Run gemini/setup.sh once first.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE")
LOG=/tmp/pupper-gemini; mkdir -p $LOG
PORT=8000
KEY_FILE=${GEMINI_KEY_FILE:-$HOME/.config/gemini-pupper/api_key}
KIOSK=0; [ "${1:-}" = "--kiosk" ] && KIOSK=1
free_port() {  # stop whatever is serving the port (an older robot server)
  local pids; pids=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oP "pid=\\K[0-9]+" | sort -u)
  [ -n "$pids" ] && kill $pids 2>/dev/null && sleep 1
  return 0
}

[ -f "$HERE/web/dist/index.html" ] || { echo "Missing $HERE/web/dist: run gemini/setup.sh first (it builds the web app)"; exit 1; }
if [ $KIOSK = 1 ] && [ ! -s "$KEY_FILE" ]; then
  echo "Missing $KEY_FILE. Run gemini/setup.sh, or create it with:  mkdir -p $(dirname $KEY_FILE) && (umask 077; cat > $KEY_FILE)"; exit 1
fi
"$ROOT/pupper/stack.sh" busy gemini/server.py || exit 1

echo "[1/3] the robot stack"
"$ROOT/pupper/stack.sh" start $LOG --assistant || exit 1

echo "[2/3] starting the robot server on localhost:$PORT (log $LOG/server.log)"
pkill -f "[g]emini/server.py" 2>/dev/null; free_port
setsid nohup bash -c "source $ROOT/pupper/ros_env.sh; cd $HERE && exec python3 -u $HERE/server.py" > $LOG/server.log 2>&1 < /dev/null &
for i in $(seq 1 30); do curl -s -o /dev/null localhost:$PORT/health && break; sleep 1; done
curl -s -o /dev/null localhost:$PORT/health || { echo "the robot server did not start; see $LOG/server.log"; tail -5 $LOG/server.log; exit 1; }

if [ $KIOSK = 0 ]; then
  echo "[3/3] ready. From a laptop:  ssh -N -L $PORT:localhost:$PORT pi@<pupper>.local   then open http://localhost:$PORT"
  exit 0
fi
echo "[3/3] opening the app on Pupper's screen"
export XDG_RUNTIME_DIR=/run/user/$(id -u) WAYLAND_DISPLAY=wayland-0
sudo -n systemctl stop pupper-gui 2>/dev/null   # the robot's own screen app
pkill -f "[p]upper-sar-kiosk" 2>/dev/null; pkill -f "[g]emini/kiosk.py" 2>/dev/null; sleep 1
amixer -q sset Master 100% 2>/dev/null; wpctl set-volume @DEFAULT_SINK@ 1.0 2>/dev/null
setsid nohup chromium --ozone-platform=wayland --kiosk --noerrdialogs --disable-infobars --no-first-run \
  --password-store=basic --use-fake-ui-for-media-stream --autoplay-policy=no-user-gesture-required \
  --remote-debugging-port=9222 --user-data-dir=$HOME/.cache/pupper-sar-kiosk http://localhost:$PORT \
  > $LOG/chromium.log 2>&1 < /dev/null &
setsid nohup python3 -u $HERE/kiosk.py "$KEY_FILE" > $LOG/kiosk.log 2>&1 < /dev/null &
sleep 12; tail -2 $LOG/kiosk.log
echo "ready. Stop with gemini/stop.sh"
