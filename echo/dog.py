"""The legs: the real Pupper, or a pretend one for the laptop.

Both have the same few methods, which is all navigator.py and server.py use:

    await dog.stand_up()            -> (ok, message)   the walking policy takes the legs
    await dog.lie_down()            -> (ok, message)   motors relaxed
    dog.send_speed(vx, vy, wz_deg)  one speed command (+vx forward, +wz counter-clockwise)
    dog.yaw_deg()                   which way the nose points, from the gyro (degrees,
                                    counter-clockwise, from wherever it was at power-on), or None
    dog.problem()                   why it can't walk right now ("" if it can)

RealDog is a thin layer over pupper/pupper_ros.py, the one file that talks to
ROS (the game player's surfer/robot_io.py is another): the controller switch,
the joystick outranking us, the e-stop, the fall detection and the heading from
the IMU are there.

SimDog walks the way a real dog does compared to its commands: a bit short, a
bit crooked, a little different every step. Its gyro drifts. The navigator is
never told where SimDog really is; only the pretend room knows (for the echoes)
and the tests (to check the map).
"""

from __future__ import annotations

import asyncio
import math
import sys
import time
from pathlib import Path

import numpy as np

PUPPER_DIR = Path(__file__).resolve().parent.parent / "pupper"
SIM_STAND_S = 1.0  # the pretend dog stands up faster than the real one (4.2 s)


class SimDog:
    mode = "sim"

    def __init__(self, x: float = 0.0, y: float = 0.0, heading_deg: float = 0.0,
                 walk_short: float = 0.85, turn_short: float = 0.9, drift_deg_per_min: float = 1.0, seed: int = 0):
        """x, y, heading_deg: where it really starts in the pretend room.
        walk_short: it really covers this share of the commanded distance.
        turn_short: ... and of the commanded turn."""
        self.x, self.y, self.heading = x, y, math.radians(heading_deg)   # the truth
        self.walk_short, self.turn_short = walk_short, turn_short
        self.drift = math.radians(drift_deg_per_min) / 60.0
        self.random = np.random.default_rng(seed)
        self.standing = False
        self.gyro_error = 0.0
        self.travelled = 0.0
        self._last: tuple[float, float, float, float] | None = None   # (time, vx, vy, wz_deg)

    async def stand_up(self) -> tuple[bool, str]:
        await asyncio.sleep(SIM_STAND_S)
        self.standing = True
        return True, "standing (simulated)"

    async def lie_down(self) -> tuple[bool, str]:
        self.standing = False
        self._last = None
        return True, "lying down (simulated)"

    def send_speed(self, vx: float, vy: float, wz_deg: float, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self._last is not None and self.standing:
            then, last_vx, last_vy, last_wz = self._last
            dt = min(0.5, now - then)   # cmd_vel_mux forgets a command after 0.5 s
            # A step is never exactly the step before: 5 % wobble, and a little sideways slip.
            wobble = 1.0 + 0.05 * self.random.standard_normal()
            forward = last_vx * self.walk_short * wobble * dt
            left = (last_vy * self.walk_short + 0.04 * last_vx * self.random.standard_normal()) * dt
            self.x += forward * math.cos(self.heading) - left * math.sin(self.heading)
            self.y += forward * math.sin(self.heading) + left * math.cos(self.heading)
            self.heading += math.radians(last_wz) * self.turn_short * wobble * dt
            self.gyro_error += self.drift * dt
            self.travelled += abs(forward) + abs(left)
        self._last = (now, vx, vy, wz_deg)

    def yaw_deg(self) -> float | None:
        return math.degrees(self.heading + self.gyro_error)

    def problem(self) -> str:
        return "" if self.standing else "lying down"


class RealDog:
    """Pupper, through pupper/pupper_ros.py. Make it inside the running asyncio loop."""

    mode = "pupper"

    def __init__(self) -> None:
        sys.path.insert(0, str(PUPPER_DIR))
        from pupper_ros import Pupper  # noqa: E402  (imports ROS: only on the robot)

        self.pupper = Pupper()  # its ROS node spins in its own thread

    async def stand_up(self) -> tuple[bool, str]:
        result = await self.pupper.stand_up()
        return result.ok, result.message

    async def lie_down(self) -> tuple[bool, str]:
        result = await self.pupper.lie_down()
        return result.ok, result.message

    def send_speed(self, vx: float, vy: float, wz_deg: float, now: float | None = None) -> None:
        self.pupper.send_speed(vx, vy, wz_deg)  # /llm_cmd_vel: the joystick still outranks it

    def yaw_deg(self) -> float | None:
        return self.pupper.yaw_deg()

    def problem(self) -> str:
        return self.pupper.problem()
