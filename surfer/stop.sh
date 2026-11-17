#!/bin/bash
# Stop Pupper playing, relax the motors and shut the robot stack down.
#   surfer/stop.sh           everything
#   surfer/stop.sh --player  only the player; leave the robot stack running
ROOT=$(cd "$(dirname "$0")/.." && pwd)
curl -s -m 2 -X POST localhost:8770/stop >/dev/null 2>&1   # stand still first
if [ "${1:-}" = "--player" ]; then
  pkill -f "[s]urfer/server.py" 2>/dev/null
  echo "player stopped (Pupper is still standing; the robot stack is still running)"; exit 0
fi
curl -s -m 5 -X POST localhost:8770/lie_down >/dev/null 2>&1 && sleep 2   # relax the motors
pkill -f "[s]urfer/server.py" 2>/dev/null
"$ROOT/pupper/stack.sh" stop
echo "stopped"
