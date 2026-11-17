"""Walking: check_move (is a move allowed?), walk (do it), and the turn, move and arc tools.

    "walk forward" -> move -> walk -> check_move -> the job, later

Every limit Gemini reads in a tool description is the same number check_move
enforces (robot_api.MoveLimits), so Gemini can plan moves the robot will accept,
and a refusal names the limit, so Gemini can retry with one that works.
"""

from __future__ import annotations

import math

from basics import PREPARES_MOVE
from following import prepare_to_move
from robot_api import FELL_MESSAGE, LIMITS, LLM_CMD_RATE, LOCOMOTION, LOCOMOTION_CONTROLLER, MoveLimits, Result, Robot, with_note
from tool_registry import Param, tool

# "Run" means at least this fast, in m/s (the walking policy switches to its
# running gait above about 0.5 m/s).
RUN_MIN_SPEED = 1.0
RUN_DEFAULT_SPEED = 1.2
DEFAULT_MOVE_S = 1.5
# The default walking and turning speeds come from robot.speeds(): 0.4 m/s and
# 60 deg/s normally, and a walking trick's own (slower) inside one.


# ============================================================ what the robot does
def check_move(vx: float, vy: float, wz_deg: float, duration: float, limits: MoveLimits = LIMITS) -> str | None:
    """None if the robot may make this move, else the reason it may not, with the limit in it.

    vx forward m/s, vy left m/s, wz_deg counter-clockwise deg/s, duration in s.
    A reason is read by Gemini word for word: "too fast" leaves it guessing;
    "too fast: |vx|<=1.5 m/s" lets it retry at a speed that works.
    """
    if abs(vx) > limits.vx_max or abs(vy) > limits.vy_max or abs(wz_deg) > limits.wz_max:
        return (f"too fast: limits are |vx|<={limits.vx_max} m/s forward or backward, |vy|<={limits.vy_max} m/s"
                f" sideways and |wz|<={limits.wz_max} deg/s turning")
    if abs(vx) < limits.vx_min and abs(vy) < limits.vy_min and abs(wz_deg) < limits.wz_min:
        # Only when all three are too small: a turn in place has vx = vy = 0, and that is fine.
        return (f"too slow to move at all: needs |vx|>={limits.vx_min} m/s, |vy|>={limits.vy_min} m/s"
                f" or |wz|>={limits.wz_min} deg/s")
    if duration <= 0 or duration > limits.duration_max:
        return f"duration must be more than 0 and at most {limits.duration_max} s (got {duration:g} s)"
    distance = math.hypot(vx, vy) * duration
    if distance > limits.distance_max:
        return (f"that would travel {distance:.2f} m, more than {limits.distance_max} m in one move:"
                " split it into shorter moves")
    return None


async def walk(robot: Robot, vx: float, vy: float, wz_deg: float, duration: float) -> Result:
    """Walk at (vx, vy, wz_deg) for duration seconds, then stand still.

    vx forward m/s, vy left m/s, wz_deg counter-clockwise deg/s. Two things to notice:

    - Why send the speed again and again? The walking policy doesn't take "walk
      for 2 seconds"; it takes a speed, over and over. In between sits
      cmd_vel_mux (the node that lets the joystick win), and it forgets a speed
      it hasn't heard for 0.5 s, so a crashed program can't leave Pupper walking
      forever. So the job sends it LLM_CMD_RATE (20) times a second for the
      whole walk, then a zero to stand still.
    - Why check robot.fallen three times? The robot may already have fallen when
      Gemini asks (1), may fall while this job waits in line (2), or while it
      walks (3). Each time, Gemini should hear FELL_MESSAGE, not "moved".

    Who walks, and how fast, depends on the state Pupper will be in when the job
    runs (robot.state_after_queue): usually the walking policy, but inside a
    trick that walks in its own stance, that trick's walking policy, with its
    own limits (robot.speeds()).
    """
    speeds = robot.speeds()  # the limits where Pupper will be (a walking trick has its own)
    refused = check_move(vx, vy, wz_deg, duration, speeds.limits)
    if refused:
        return Result(False, refused + speeds.where)  # Gemini hears why; nothing is queued
    note = await prepare_to_move(robot)
    state = robot.state_after_queue
    walker = robot.walking_controller(state)  # the walking policy, or a walking trick's own
    then = state if walker != LOCOMOTION_CONTROLLER else LOCOMOTION  # a walking trick stays in its stance
    if robot.fallen:  # (1)
        return Result(False, FELL_MESSAGE)

    async def job() -> Result:
        if robot.fallen:  # (2)
            return Result(False, FELL_MESSAGE)
        res = await robot.switch_to(walker)  # the policy that walks takes the legs (if it hasn't already)
        if not res.ok:
            return res
        end = robot.now() + duration
        while robot.now() < end - 1e-6:  # keep the mux input alive (1e-6: no extra step from rounding)
            if robot.fallen:  # (3)
                robot.send_speed(0, 0, 0)
                return Result(False, FELL_MESSAGE)
            robot.send_speed(vx, vy, wz_deg)
            await robot.sleep(1.0 / LLM_CMD_RATE)  # never time.sleep: it would freeze the server
        robot.send_speed(0, 0, 0)  # stand still
        return Result(True, f"moved {what} for {duration:.2f} s")  # the outcome (debug log)

    what = f"vx={vx:g} vy={vy:g} wz={wz_deg:.3g}"  # 45.836623610465864 -> 45.8: Gemini reads this
    return with_note(note, await robot.queue(f"move {what} {duration:.3g}s", job, duration, then=then))


async def turn_in_place(robot: Robot, direction: str, angle: float) -> Result:
    """Turn left (counter-clockwise, +wz) or right by angle degrees, at the turning speed where Pupper will be."""
    turn_speed = robot.speeds().turn
    wz_deg = turn_speed if direction == "left" else -turn_speed
    return await walk(robot, 0.0, 0.0, wz_deg, abs(angle) / turn_speed)


# ============================================================ what Gemini reads
@tool(
    "turn",
    "Turn in place to the left or right by an angle in degrees, at about 60 deg/s (slower inside a"
    f" trick that walks). At most {LIMITS.duration_max:g} s of turning in one call." + PREPARES_MOVE,
    direction=Param("string", "Which way to turn: left (counter-clockwise) or right (clockwise).", enum=["left", "right"]),
    angle=Param("number", "How far to turn, in degrees, e.g. 90 for a quarter turn, 180 to face the other way."
                " Default 90.", required=False),
)
async def turn_tool(robot: Robot, direction: str, angle: float = 90.0) -> Result:
    return await turn_in_place(robot, direction, angle)


@tool(
    "move",
    "Walk or run in a straight line: forward, backward, left or right (sideways), for a number of"
    f" seconds. Walking is 0.4 m/s by default (at least {LIMITS.vx_min} m/s); running is forward or"
    f" backward only, {RUN_MIN_SPEED}-{LIMITS.vx_max} m/s. Sideways is at most {LIMITS.vy_max} m/s and"
    f" never runs. One move goes at most {LIMITS.distance_max} m and {LIMITS.duration_max:g} s; split"
    " longer trips into several moves." + PREPARES_MOVE,
    direction=Param("string", "Which way to go: forward, backward, left or right (sideways).",
                    enum=["forward", "backward", "left", "right"]),
    run=Param("boolean", f"true to run (forward or backward only, at least {RUN_MIN_SPEED} m/s). Default false.",
              required=False),
    speed=Param("number", f"Speed in m/s. Default 0.4 walking, {RUN_DEFAULT_SPEED} running.", required=False),
    duration=Param("number", f"How long to move, in seconds. Default {DEFAULT_MOVE_S:g}.", required=False),
)
async def move_tool(robot: Robot, direction: str, run: bool = False, speed: float | None = None,
                    duration: float = DEFAULT_MOVE_S) -> Result:
    straight = direction in ("forward", "backward")
    if run and straight:
        speed = max(RUN_DEFAULT_SPEED if speed is None else speed, RUN_MIN_SPEED)
    elif speed is None:
        speed = robot.speeds().walk
    vx = {"forward": speed, "backward": -speed}.get(direction, 0.0)
    vy = {"left": speed, "right": -speed}.get(direction, 0.0)  # the robot's left is +y
    note = "Sideways never runs; walking instead." if run and not straight else ""
    return with_note(note, await walk(robot, vx, vy, 0.0, duration))


@tool(
    "arc",
    "Walk forward and turn at the same time, along part of a circle: e.g. angle 360 for a whole circle,"
    " 90 for a curved quarter turn. radius is the circle's size in metres; radius 0 is a sharp corner"
    f" (a turn in place). The robot turns at most {LIMITS.wz_max} deg/s, so at speed v the radius must"
    f" be at least v / {math.radians(LIMITS.wz_max):.2f} (0.2 m at the default 0.4 m/s). One arc goes at"
    f" most {LIMITS.distance_max} m; split a bigger circle into several arcs." + PREPARES_MOVE,
    direction=Param("string", "Which way to curve: left (counter-clockwise) or right (clockwise).", enum=["left", "right"]),
    angle=Param("number", "How much of the circle to walk, in degrees: 90 a quarter, 180 a half, 360 all the way round."),
    radius=Param("number", "The circle's radius in metres. Default 0.5. 0 means turn in place (a sharp corner).",
                 required=False),
    speed=Param("number", "Walking speed in m/s. Default 0.4.", required=False),
)
async def arc_tool(robot: Robot, direction: str, angle: float, radius: float = 0.5, speed: float | None = None) -> Result:
    speed = robot.speeds().walk if speed is None else speed
    if radius < 0 or speed <= 0:
        return Result(False, f"no arc: radius must be 0 m or more and speed more than 0 m/s (got radius {radius:g} m,"
                             f" speed {speed:g} m/s)")
    if radius == 0:  # a sharp corner: Gemini plans whole shapes and asks for corners this way
        return with_note("Radius 0: turned in place (a sharp corner).", await turn_in_place(robot, direction, angle))
    wz_max = robot.speeds().limits.wz_max
    wz_deg = math.degrees(speed / radius)
    if wz_deg > wz_max:
        smallest = math.ceil(speed / math.radians(wz_max) * 100) / 100
        return Result(False, f"too tight: at {speed:g} m/s the radius must be at least {smallest} m"
                             " (or walk slower)")
    wz_deg = wz_deg if direction == "left" else -wz_deg
    return await walk(robot, speed, 0.0, wz_deg, abs(angle) / abs(wz_deg))
