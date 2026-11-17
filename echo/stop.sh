#!/bin/bash
# Stop Pupper walking by ear, relax the motors and shut the robot stack down.
#   echo/stop.sh           everything
#   echo/stop.sh --ears    only the ears' server; leave the robot stack running
ROOT=$(cd "$(dirname "$0")/.." && pwd)
curl -s -m 2 -X POST localhost:8771/stop >/dev/null 2>&1   # stand still first
if [ "${1:-}" = "--ears" ]; then
  pkill -f "[e]cho/server.py" 2>/dev/null
  echo "ears stopped (Pupper is still standing; the robot stack is still running)"; exit 0
fi
curl -s -m 5 -X POST localhost:8771/lie_down >/dev/null 2>&1 && sleep 2   # relax the motors
pkill -f "[e]cho/server.py" 2>/dev/null
"$ROOT/pupper/stack.sh" stop
echo "stopped"
