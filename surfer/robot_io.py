"""The robot behind the moves: the real Pupper, or a pretend one for the laptop.

Both have the same few methods, which is all moves.py and server.py use:

    await robot.stand_up()            -> (ok, message)   the walking policy takes the legs
    await robot.lie_down()            -> (ok, message)   motors relaxed
    robot.send_speed(vx, vy, wz_deg)  one speed command (+vy left, +wz counter-clockwise)
    robot.play_reference(name)        send the "jump" or "squat" table to the walking policy
    robot.reference_seconds(name)     how long that table plays
    robot.frame()                     the newest camera picture (BGR), or None
    robot.problem()                   why it can't play right now ("" if it can)

RealRobot is a thin layer over pupper/pupper_ros.py, the one file that talks to
ROS: the controller switch, the joystick outranking us, the e-stop and the fall
detection are there. The robot stack must be running first (surfer/start.sh
does both). SimRobot needs nothing but Python, so the whole game loop can be
developed and tested on a laptop.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PUPPER_DIR = HERE.parent / "pupper"
REFERENCES = {
    "jump": PUPPER_DIR / "references" / "jump.json",   # the table the walking policy was trained on
    "squat": HERE / "references" / "squat.json",       # ours: squat_reference.py
}
SIM_STAND_S = 1.0  # the pretend robot stands up faster than the real one (4.2 s)


def load_reference(path: Path) -> tuple[float, list[float]]:
    """A reference file -> (seconds it plays, the message for the walking policy).

    The message is playback_s, active_s, cross_fade_s, then every row of the
    table (12 joint angles each), flat.
    """
    ref = json.loads(path.read_text())
    data = [float(ref["playback_s"]), float(ref["active_s"]), float(ref["cross_fade_s"])]
    for row in ref["table"]:
        if len(row) != 12:
            raise ValueError(f"{path}: every table row needs 12 joint angles")
        data.extend(float(v) for v in row)
    return float(ref["playback_s"]), data


class SimRobot:
    """A pretend Pupper: remembers what it was told, moves nothing.

    It keeps a rough idea of where the speed commands would have taken it
    (self.left, in metres), which the tests use to check the moves.
    """

    mode = "sim"

    def __init__(self) -> None:
        self.standing = False
        self.left = 0.0
        self.commands: list[tuple[float, float, float, float]] = []  # (time, vx, vy, wz_deg)
        self.references: list[tuple[float, str]] = []                # (time, name)
        self.seconds = {name: load_reference(path)[0] for name, path in REFERENCES.items()}
        self._last = None

    async def stand_up(self) -> tuple[bool, str]:
        await asyncio.sleep(SIM_STAND_S)
        self.standing = True
        return True, "standing (simulated)"

    async def lie_down(self) -> tuple[bool, str]:
        self.standing = False
        return True, "lying down (simulated)"

    def send_speed(self, vx: float, vy: float, wz_deg: float, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self._last is not None:
            last_time, last_vy = self._last
            self.left += last_vy * min(0.2, now - last_time)
        self._last = (now, vy)
        self.commands.append((now, vx, vy, wz_deg))
        del self.commands[:-2000]

    def play_reference(self, name: str) -> None:
        self.references.append((time.monotonic(), name))

    def reference_seconds(self, name: str) -> float:
        return self.seconds[name]

    def frame(self) -> np.ndarray | None:
        return None  # no camera: the laptop streams the game view instead

    def problem(self) -> str:
        return "" if self.standing else "lying down"


class RealRobot:
    """Pupper, through pupper/pupper_ros.py. Make it inside the running asyncio loop."""

    mode = "pupper"

    def __init__(self) -> None:
        sys.path.insert(0, str(PUPPER_DIR))
        from pupper_ros import Pupper  # noqa: E402  (imports ROS: only on the robot)

        self.pupper = Pupper()  # its ROS node spins in its own thread
        self.messages, self.seconds = {}, {}
        for name, path in REFERENCES.items():
            self.seconds[name], self.messages[name] = load_reference(path)

    async def stand_up(self) -> tuple[bool, str]:
        # Hands the legs to the walking policy and waits until it has faded in;
        # after a fall it first checks that Pupper is upright and still.
        result = await self.pupper.stand_up()
        return result.ok, result.message

    async def lie_down(self) -> tuple[bool, str]:
        result = await self.pupper.lie_down()
        return result.ok, result.message

    def send_speed(self, vx: float, vy: float, wz_deg: float) -> None:
        self.pupper.send_speed(vx, vy, wz_deg)  # /llm_cmd_vel: the joystick still outranks it

    def play_reference(self, name: str) -> None:
        self.pupper.play_reference(self.messages[name])  # /sar_walk/reference_trick

    def reference_seconds(self, name: str) -> float:
        return self.seconds[name]

    def frame(self) -> np.ndarray | None:
        jpeg = self.pupper.camera_jpeg()  # /camera/image_raw/compressed
        if jpeg is None:
            return None
        return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)

    def problem(self) -> str:
        return self.pupper.problem()
