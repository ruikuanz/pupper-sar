"""The five moves, on the dog.

    game move (phone)      Pupper
    ---------------------  ---------------------------------------------------
    nothing                RUN      run in place, in one of three spots
    swipe left / right     SHUFFLE  sidestep one lane (LANE_SPACING_M)
    swipe up               JUMP     the jump the walking policy was trained on
    swipe down             SQUAT    a crouch, played the same way as the jump

Everything goes through the walking policy (the sar_walk controller,
pupper/walk_controller.py), which is what makes the moves fast enough for a
game: no controller switch, so no 4-second stand-up between moves.

  - RUN and SHUFFLE are speed commands. With no forward speed and a
    command of at least 0.1, the policy plays the *lift gait*: it steps in
    place. So "running in place" is a small sideways speed that flips sign a
    couple of times a second (RUN_VY, RUN_SWITCH_S): the feet keep stepping and
    the body stays put. A shuffle is the same gait with a real sideways speed.
  - JUMP and SQUAT are *reference tricks*: a table of
    joint angles sent once to the running policy, which follows it and keeps
    its balance. The jump table is the one the policy was trained on. The squat
    table is ours (squat_reference.py); see the warning there.

Surfer is a state machine with a clock: tick() runs
20 times a second, sends one speed command, and moves on when a phase is over.
It never sleeps, so it never blocks the server (or the stop button).

Signs are the robot's: +vy is LEFT, +wz is counter-clockwise (a left turn).
Lanes are the game's: 0 left, 1 centre, 2 right.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from policy import Timing

# --- the three spots ----------------------------------------------------------
LANE_SPACING_M = 0.25   # m between two lanes on the floor
SHUFFLE_VY = 0.40       # m/s sideways during a shuffle (the policy's cap is 0.5)
SHUFFLE_S = 0.80        # s one shuffle takes: LANE_SPACING_M / SHUFFLE_VY, plus getting going
# Tune SHUFFLE_S on the floor: put tape at the three spots, shuffle left and
# right a few times, and change it until Pupper lands on the tape.

# --- running in place ---------------------------------------------------------
RUN_VY = 0.12           # m/s: just over the 0.1 where the lift gait is fully on
RUN_SWITCH_S = 0.4      # s between sign flips: left, right, left, ... adds up to nowhere

# --- jump and squat -----------------------------------------------------------
# Stand still for a moment, send the table, then leave the policy alone while it
# plays and for a moment after. The controller refuses a new table for 1.0 s
# after the last one started, so playback + recover + settle must stay above 1.0.
JUMP_SETTLE_S = 0.25
JUMP_RECOVER_S = 0.30
SQUAT_SETTLE_S = 0.10
SQUAT_RECOVER_S = 0.20

# --- holding the spot with the camera (off unless --hold) ----------------------
# When Pupper sees the game's frame it knows where it stands (screen_finder.py).
# Then a shuffle ends where the lane really is, not after a fixed time, and
# running in place steers back to the spot.
KP_LANE = 2.0           # m/s of sideways speed per m of error
HOLD_VY_MAX = 0.30      # m/s
LANE_TOLERANCE_M = 0.04
KP_DISTANCE = 1.0       # m/s forward per m too far from the screen
HOLD_VX_MAX = 0.15
DISTANCE_TOLERANCE_M = 0.06
KP_YAW = 1.0            # deg/s of turn per degree of facing error
YAW_DEADBAND_DEG = 6.0
YAW_MAX_DEG_S = 25.0
SHUFFLE_TIMEOUT = 1.8   # x SHUFFLE_S: give up steering to the lane after this long


class State(Enum):
    IDLE = 0      # not playing: silent, so the joystick still works
    RUN = 1       # running in place, ready for a move
    SHUFFLE = 2   # changing lane
    JUMP = 3
    SQUAT = 4


@dataclass
class Position:
    """Where Pupper stands, from the game's frame in its camera."""

    left: float       # m to the left of the frame's centre line (+ is Pupper's left)
    distance: float   # m in front of the screen
    facing_right_deg: float  # degrees Pupper is turned clockwise away from square-on


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class Surfer:
    def __init__(self, robot, emit: Callable[[dict], None], hold: bool = False):
        """robot: send_speed(vx, vy, wz_deg), play_reference(name) and
        reference_seconds(name) (robot_io.py). emit: called with each move's
        message for the game, the moment the move starts. hold: steer to the
        lane with the camera (Position) instead of by the clock."""
        self.robot = robot
        self.emit = emit
        self.hold = hold
        self.state = State.IDLE
        self.lane = 1
        self.since = 0.0          # when the current state began
        self.sent_table = False   # jump/squat: has the table been sent yet?
        self.direction = 0        # shuffle: +1 left, -1 right
        self.came_from = 1        # shuffle: the lane it left
        self.home: Position | None = None  # where the centre lane is (set when a game starts)

    # -------------------------------------------------------------- timing
    def timing(self) -> Timing:
        """How long the moves take, for the policy: when must it decide?"""
        return Timing(
            shuffle_s=SHUFFLE_S,
            jump_lead_s=JUMP_SETTLE_S + 0.5 * self.robot.reference_seconds("jump"),
            squat_lead_s=SQUAT_SETTLE_S + 0.5 * self.robot.reference_seconds("squat"),
        )

    # -------------------------------------------------------------- start, stop
    def start(self, now: float) -> None:
        """Begin a game: this spot is the centre lane."""
        self.state, self.since, self.lane, self.home = State.RUN, now, 1, None

    def stop(self) -> None:
        """Stand still. (One zero command: after that, IDLE says nothing.)"""
        if self.state != State.IDLE:
            self.robot.send_speed(0.0, 0.0, 0.0)
        self.state = State.IDLE

    @property
    def lanes(self) -> tuple[int, ...]:
        """The lane(s) the runner on screen is in: two of them during a shuffle."""
        return (self.came_from, self.lane) if self.state == State.SHUFFLE else (self.lane,)

    @property
    def free(self) -> bool:
        """Can a new move start right now?"""
        return self.state == State.RUN

    # -------------------------------------------------------------- the moves
    def begin(self, move: str, now: float) -> bool:
        """Start one of the moves. Returns False if it can't start (busy, or no lane there)."""
        if not self.free or move == "run":
            return False
        if move in ("left", "right"):
            target = self.lane + (-1 if move == "left" else 1)
            if not 0 <= target <= 2:
                return False
            self.came_from, self.lane = self.lane, target
            self.direction = 1 if move == "left" else -1
            self.state = State.SHUFFLE
            # The game's runner starts across now, and takes as long as the dog does.
            self.emit({"type": "move", "move": move, "duration": SHUFFLE_S, "lane": self.lane})
        elif move in ("jump", "squat"):
            self.state = State.JUMP if move == "jump" else State.SQUAT
            self.sent_table = False
        else:
            return False
        self.since = now
        return True

    def target_left(self) -> float:
        """Where the current lane is, in m left of the screen's centre line."""
        home = self.home.left if self.home else 0.0
        return home + (1 - self.lane) * LANE_SPACING_M

    def steer(self, position: Position | None) -> tuple[float, float, float] | None:
        """(vx, vy, wz_deg) that brings Pupper back to its spot, or None without a position."""
        if not self.hold or position is None:
            return None
        if self.home is None:
            self.home = position
        error = self.target_left() - position.left
        vy = clamp(KP_LANE * error, HOLD_VY_MAX) if abs(error) > LANE_TOLERANCE_M else 0.0
        too_far = position.distance - self.home.distance
        vx = clamp(KP_DISTANCE * too_far, HOLD_VX_MAX) if abs(too_far) > DISTANCE_TOLERANCE_M else 0.0
        facing = position.facing_right_deg
        wz = clamp(KP_YAW * facing, YAW_MAX_DEG_S) if abs(facing) > YAW_DEADBAND_DEG else 0.0
        return vx, vy, wz

    def tick(self, now: float, position: Position | None = None) -> None:
        """Run 20 times a second: send this moment's speed command."""
        if self.state == State.IDLE:
            return
        elapsed = now - self.since
        vx = vy = wz = 0.0
        steer = self.steer(position)

        if self.state == State.RUN:
            # Left, right, left, right: stepping without going anywhere.
            vy = RUN_VY if int(elapsed / RUN_SWITCH_S) % 2 == 0 else -RUN_VY
            if steer and any(steer):
                vx, vy, wz = steer

        elif self.state == State.SHUFFLE:
            if steer:
                error = self.target_left() - position.left
                done = abs(error) <= LANE_TOLERANCE_M and elapsed >= 0.5 * SHUFFLE_S
                vx, vy, wz = steer[0], clamp(KP_LANE * error * 2.0, SHUFFLE_VY), steer[2]
                if done or elapsed >= SHUFFLE_TIMEOUT * SHUFFLE_S:
                    self.state, self.since = State.RUN, now
            else:
                vy = self.direction * SHUFFLE_VY
                if elapsed >= SHUFFLE_S:
                    self.state, self.since = State.RUN, now
                    vy = 0.0

        else:  # JUMP or SQUAT: stand still, send the table once, wait for it
            name = "jump" if self.state == State.JUMP else "squat"
            settle = JUMP_SETTLE_S if name == "jump" else SQUAT_SETTLE_S
            recover = JUMP_RECOVER_S if name == "jump" else SQUAT_RECOVER_S
            seconds = self.robot.reference_seconds(name)
            if not self.sent_table and elapsed >= settle:
                self.robot.play_reference(name)
                self.sent_table = True
                self.emit({"type": "move", "move": name, "duration": seconds, "lane": self.lane})
            if elapsed >= settle + seconds + recover:
                self.state, self.since = State.RUN, now

        self.robot.send_speed(vx, vy, wz)
