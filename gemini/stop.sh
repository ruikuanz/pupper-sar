#!/bin/bash
# Stop the assistant: relax the motors, stop the app, the server and the robot stack,
# and bring back the robot's own screen app.
#   gemini/stop.sh             everything
#   gemini/stop.sh --server    only the server and the kiosk; leave the robot stack running
ROOT=$(cd "$(dirname "$0")/.." && pwd)
curl -s -m 3 -X POST localhost:8000/stop/immediate >/dev/null 2>&1
curl -s -m 3 -X POST localhost:8000/walking/deactivate >/dev/null 2>&1 && sleep 2   # relax the motors
pkill -f "[g]emini/kiosk.py"; pkill -f "[p]upper-sar-kiosk"; pkill -f "[g]emini/server.py"
if [ "${1:-}" != "--server" ]; then
  "$ROOT/pupper/stack.sh" stop
fi
sudo -n systemctl start pupper-gui 2>/dev/null
echo "stopped"
