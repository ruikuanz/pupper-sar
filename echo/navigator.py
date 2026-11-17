"""Walking blind: listen, decide, walk, listen again.

Pupper never looks. Its whole idea of the room comes from its ears:

    LOOK     stand still and ping. Four ears: one ping. Two ears: ping, turn
             60 degrees, ping, turn, ping (ears.py says why).
    DECIDE   put the echoes on the map, correct the position with them
             (room_map.py), and choose where to step.
    TURN     face that way (the gyro says when).
    WALK     a step of at most STEP_M, never further than the echoes say is
             free, then look again.

Three things it can be asked to do:

    look      one look, no walking: "what is around me?"
    explore   keep stepping toward places it has not been, mapping as it goes
    goto      walk to a point on the map, around what is in the way

It is a state machine with a clock, like the game player's
surfer/moves.py: tick() runs 20 times a second, sends one speed command and never
waits. (The ping takes half a second, so it runs in a helper thread; tick just
checks whether it is back.)

Pupper stands still to listen: its own motors and feet are loud, and a chirp
heard while the body moves is smeared. Walk, stop, listen.

Signs are the robot's: +vx forward, +wz counter-clockwise. Angles in the room
frame are counter-clockwise from where the nose pointed at the start.
"""

from __future__ import annotations

import math
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from typing import Callable

import numpy as np

from ears import Ears, Echo, Hearing
from room_map import RoomMap

# --- walking (tune these on the floor, like SHUFFLE_S in surfer/moves.py) -------------
WALK_SPEED = 0.30        # m/s commanded. Below 0.2 the policy barely moves
WALK_SCALE = 1.0         # metres really covered per metre commanded. Measure it: "Walk 1 m"
#                          on the map page, a tape measure, and put the result here
STEP_M = 0.5             # the longest step between two looks
MIN_STEP_M = 0.15        # shorter than this is not worth a step
HOLD_KP = 2.0            # deg/s of turn per degree off course, while walking

# --- turning ----------------------------------------------------------------------
TURN_SPEED = 45.0        # deg/s at most
TURN_SLOWEST = 20.0      # deg/s at least: slower commands barely turn
TURN_KP = 2.5            # deg/s per degree still to go
TURN_CLOSE_DEG = 4.0     # near enough
TURN_SCALE = 1.0         # only used without a gyro: degrees really turned per degree commanded
TURN_TIMEOUT_S = 12.0

# --- listening --------------------------------------------------------------------
SETTLE_S = 0.5           # stand this long before a ping: let the legs go quiet
LOOK_TURNS_DEG = (0.0, 60.0, 120.0)   # two ears: the headings of one look, from where it started

# --- staying off the walls ----------------------------------------------------------
BODY_M = 0.30            # the middle of the body stays this far from anything (half a dog + a margin)
SLACK_M = 0.10           # ... plus this, because a step never lands exactly
FEELERS_DEG = (-15.0, 0.0, 15.0)   # a direction is as free as the tightest of these: the body has width
TIGHT_M = 0.5            # walls this near on two sides: too tight. Their echoes pile up on each other and
#                          the distances can no longer be trusted, so Pupper does not step at all

# --- the jobs -----------------------------------------------------------------------
EXPLORE_LOOKS = 8        # explore stops after this many looks
GOTO_LOOKS = 20          # goto gives up after this many
ARRIVE_M = 0.15          # near enough to the goal


class State(Enum):
    IDLE = 0      # not doing anything: silent, so the joystick still works
    TURN = 1      # turning to self.aim_deg
    SETTLE = 2    # standing still before a ping
    LISTEN = 3    # the ping is playing; waiting for the recording
    WALK = 4      # walking self.step_m along self.aim_deg


def wrap(deg: float) -> float:
    """An angle folded into -180..180."""
    return (deg + 180.0) % 360.0 - 180.0


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class Navigator:
    def __init__(self, dog, sound, ears: Ears, say: Callable[[str], None] = print, background: bool = True):
        """dog: send_speed, yaw_deg, problem (dog.py). sound: ping() (sound.py).
        say: called with each line of the log. background: run pings in a
        helper thread (the server); False runs them on the spot (the tests)."""
        self.dog, self.sound, self.ears, self.say = dog, sound, ears, say
        self.pool = ThreadPoolExecutor(max_workers=1) if background else None
        self.map = RoomMap()
        self.state = State.IDLE
        self.job = ""                 # "", "look", "explore" or "goto"
        self.goal: np.ndarray | None = None
        self.message = "ready"
        # Where Pupper thinks it is, in the room frame.
        self.x = self.y = 0.0
        self.heading = 0.0            # deg; from the gyro when there is one
        self.yaw_zero: float | None = None
        self.trail: list[tuple[float, float]] = [(0.0, 0.0)]
        # The look in progress.
        self.to_ping: list[float] = []            # headings still to ping at (deg, room frame)
        self.pings: list[tuple[float, Hearing]] = []
        self.pending: Future | None = None
        self.look_side = 1.0          # two ears: turn left on one look, right on the next
        self.looks_this_job = 0
        self.detour = 0.0
        self.echoes: list[Echo] = []  # the last look's
        self.rings: list[Echo] = []   # ... its doubtful peaks: off the map, but not to be walked into
        self.dropped: list[Echo] = []
        self.hearing: Hearing | None = None       # the last ping's, for the map page
        self.problems: list[str] = []
        # The move in progress.
        self.aim_deg = 0.0
        self.step_m = 0.0
        self.walked_m = 0.0
        self.since = 0.0
        self.last_tick: float | None = None

    # ---------------------------------------------------------------- start, stop
    def start(self, job: str, now: float, goal: tuple[float, float] | None = None) -> str:
        """Begin a job. Returns "" if it started, or why it can't."""
        if job not in ("look", "explore", "goto"):
            return f"unknown job '{job}'"
        if job == "goto" and goal is None:
            return "goto needs a point"
        problem = self.dog.problem()
        if problem and self.must_stand(job):
            return f"can't do that: {problem}"
        self.job, self.goal = job, (np.array(goal, float) if goal is not None else None)
        self.looks_this_job = 0
        self.detour = 0.0             # goto: +1 going round to the left of something, -1 to the right
        self.say(f"{job}" + (f" ({goal[0]:+.2f}, {goal[1]:+.2f})" if goal is not None else ""))
        self.begin_look(now)
        return ""

    def walk_test(self, now: float, metres: float = 1.0) -> str:
        """Walk straight ahead without listening, to measure WALK_SCALE with a tape. Needs clear floor."""
        problem = self.dog.problem()
        if problem:
            return f"can't do that: {problem}"
        if not 0.2 <= metres <= 1.5:
            return "the walk test goes 0.2 to 1.5 m"
        self.job, self.aim_deg, self.step_m, self.walked_m = "walk_test", self.heading, metres, 0.0
        self.to_ping = []
        self.state, self.since = State.WALK, now
        self.message = f"walk test: {metres:.2f} m straight ahead, NOT listening"
        self.say(self.message)
        return ""

    def stop(self, why: str = "stopped") -> None:
        """Stand still. (One zero command: after that, IDLE says nothing.)"""
        if self.state != State.IDLE:
            self.dog.send_speed(0.0, 0.0, 0.0)
            self.say(why)
        self.state, self.job, self.message = State.IDLE, "", why
        self.to_ping, self.pings, self.pending = [], [], None

    def forget(self) -> None:
        """A fresh map; here and now become the origin."""
        self.stop("map cleared")
        self.map = RoomMap()
        self.x = self.y = 0.0
        self.yaw_zero = None
        self.heading = 0.0
        self.trail = [(0.0, 0.0)]
        self.echoes, self.rings, self.dropped = [], [], []

    @property
    def busy(self) -> bool:
        return self.state != State.IDLE

    def must_stand(self, job: str) -> bool:
        """Everything needs the legs, except a look with four ears (no turning): that works lying down."""
        return job != "look" or self.ears.geometry.needs_turning

    # ---------------------------------------------------------------- where am I pointing?
    def read_heading(self, wz_deg: float, dt: float) -> None:
        """Update self.heading: from the gyro, or failing that by counting the turn commanded."""
        yaw = self.dog.yaw_deg()
        if yaw is None:
            self.heading += wz_deg * TURN_SCALE * dt
            return
        if self.yaw_zero is None:
            self.yaw_zero = yaw - self.heading
        self.heading = yaw - self.yaw_zero

    # ---------------------------------------------------------------- the look
    def begin_look(self, now: float) -> None:
        turns = LOOK_TURNS_DEG if self.ears.geometry.needs_turning else (0.0,)
        self.to_ping = [self.heading + self.look_side * t for t in turns]
        self.look_side = -self.look_side
        self.pings = []
        self.next_ping(now)

    def next_ping(self, now: float) -> None:
        self.aim_deg = self.to_ping[0]
        self.state, self.since = State.TURN, now
        self.message = "listening"

    def finish_look(self, now: float) -> None:
        """All pings are in: find the echoes, fix the position, map them, decide."""
        look = self.ears.locate(self.pings)
        self.echoes, self.rings, self.dropped = look.echoes, look.rings, look.dropped
        before = np.array([self.x, self.y])
        spot, matched = self.map.fix(before, look.echoes, step=before - np.array(self.trail[-1]))
        self.x, self.y = float(spot[0]), float(spot[1])
        self.map.add(spot, look.echoes)
        self.trail.append((self.x, self.y))
        self.looks_this_job += 1
        moved = float(np.linalg.norm(spot - before))
        nearest = min((e.distance_m for e in look.echoes), default=float("inf"))
        self.say(f"look {self.map.looks}: {len(look.echoes)} echoes, nearest {nearest:.2f} m; "
                 f"{matched} matched the map, position corrected {moved * 100:.0f} cm")
        self.decide(now)

    # ---------------------------------------------------------------- deciding
    def free(self, direction_deg: float) -> float:
        """How far Pupper can step in a direction and still be BODY_M from everything."""
        spot = np.array([self.x, self.y])
        heard = self.echoes + self.rings   # when it comes to not hitting things, a doubtful echo counts
        return min(self.map.clearance(spot, direction_deg + f, heard) for f in FEELERS_DEG) - BODY_M

    def decide(self, now: float) -> None:
        if self.job == "look" or not self.job:
            return self.stop(f"looked: {len(self.echoes)} echoes")
        here = np.array([self.x, self.y])
        near = [e for e in self.echoes if e.distance_m < TIGHT_M]
        if any(abs(wrap(a.bearing_deg - b.bearing_deg)) >= 90.0 for a in near for b in near):
            return self.stop(f"too tight to walk by ear: walls within {TIGHT_M} m on more than one side")

        if self.job == "explore":
            if self.looks_this_job >= EXPLORE_LOOKS:
                return self.stop(f"explored: {self.map.looks} looks, {sum(m.confirmed for m in self.map.landmarks)} landmarks confirmed")
            best = None
            for direction in np.arange(-180.0, 180.0, 15.0):
                step = min(STEP_M, self.free(direction) - SLACK_M)
                if step < MIN_STEP_M:
                    continue
                end = here + step * np.array([math.cos(math.radians(direction)), math.sin(math.radians(direction))])
                # Go where it has not looked from yet; prefer open directions and small turns.
                new_ground = min(float(np.linalg.norm(end - s)) for s in self.map.spots)
                score = (min(new_ground, 1.5) + 0.3 * min(self.free(direction), 2.0)
                         - 0.2 * abs(wrap(direction - self.heading)) / 180.0)
                if best is None or score > best[0]:
                    best = (score, direction, step)
            if best is None:
                return self.stop("boxed in: no direction has room for a step")
            _, self.aim_deg, self.step_m = best

        else:  # goto
            to_goal = self.goal - here
            distance = float(np.linalg.norm(to_goal))
            if distance < ARRIVE_M:
                return self.stop(f"arrived at ({self.goal[0]:+.2f}, {self.goal[1]:+.2f}), {distance * 100:.0f} cm off")
            if self.looks_this_job >= GOTO_LOOKS:
                return self.stop(f"gave up: still {distance:.2f} m from the goal")
            straight = math.degrees(math.atan2(to_goal[1], to_goal[0]))
            best = None
            for aside in (0.0, 20.0, -20.0, 40.0, -40.0, 60.0, -60.0, 80.0, -80.0, 100.0, -100.0):
                if aside * self.detour < 0:
                    continue   # going round something: keep to the side already chosen, don't dither
                step = min(STEP_M, distance, self.free(straight + aside) - SLACK_M)
                if step < min(MIN_STEP_M, distance):
                    continue
                way = math.radians(straight + aside)
                end = here + step * np.array([math.cos(way), math.sin(way)])
                progress = distance - float(np.linalg.norm(self.goal - end))   # how much nearer the goal this step ends
                # A step must bring the goal nearer, unless it is a sidestep around what is in the way.
                if (progress > 0.03 or abs(aside) >= 60.0) and (best is None or progress > best[0] + 0.02):
                    best = (progress, straight + aside, step, aside)
            if best is None:
                return self.stop(f"blocked: no way toward the goal from here ({distance:.2f} m to go)")
            _, self.aim_deg, self.step_m, aside = best
            self.detour = 0.0 if aside == 0.0 else math.copysign(1.0, aside)

        self.walked_m = 0.0
        self.state, self.since = State.TURN, now
        self.message = f"{self.job}: stepping {self.step_m:.2f} m toward {wrap(self.aim_deg):+.0f} deg"

    # ---------------------------------------------------------------- the clock
    def tick(self, now: float) -> None:
        """Run 20 times a second: send this moment's speed command."""
        dt = 0.0 if self.last_tick is None else min(0.2, now - self.last_tick)
        self.last_tick = now
        if self.state == State.IDLE:
            self.read_heading(0.0, dt)
            return
        problem = self.dog.problem()
        if problem and self.must_stand(self.job):
            return self.stop(f"stopped: {problem}")
        vx = wz = 0.0
        elapsed = now - self.since

        if self.state == State.TURN:
            off = wrap(self.aim_deg - self.heading)
            if abs(off) <= TURN_CLOSE_DEG:
                # Facing it. A ping is next if this turn was part of a look; else the step.
                self.state, self.since = (State.SETTLE if self.to_ping else State.WALK), now
            elif elapsed > TURN_TIMEOUT_S:
                return self.stop("stopped: the turn is not getting there (is Pupper stuck against something?)")
            else:
                wz = math.copysign(max(TURN_SLOWEST, min(TURN_SPEED, TURN_KP * abs(off))), off)

        elif self.state == State.SETTLE:
            if elapsed >= SETTLE_S:
                self.pending = self.pool.submit(self.sound.ping) if self.pool else self.call_now(self.sound.ping)
                self.state, self.since = State.LISTEN, now

        elif self.state == State.LISTEN:
            if self.pending.done():
                try:
                    self.hearing = self.ears.listen(self.pending.result())
                except Exception as error:  # no device, too few channels, no chirp in the recording
                    return self.stop(f"stopped: could not listen ({error})")
                self.problems = self.hearing.problems
                for line in self.problems:
                    self.say(f"  ! {line}")
                self.pings.append((math.radians(self.heading), self.hearing))
                self.to_ping.pop(0)
                if self.to_ping:
                    self.next_ping(now)
                else:
                    self.finish_look(now)   # ends in IDLE, or in TURN toward the next step

        elif self.state == State.WALK:
            if self.walked_m >= self.step_m and self.job == "walk_test":
                return self.stop(f"walk test done: asked for {self.step_m:.2f} m. Measure how far Pupper really went; "
                                 "WALK_SCALE = measured / asked")
            if self.walked_m >= self.step_m:
                self.begin_look(now)
            else:
                vx = WALK_SPEED
                wz = clamp(HOLD_KP * wrap(self.aim_deg - self.heading), TURN_SLOWEST)
                self.walked_m += WALK_SPEED * WALK_SCALE * dt
                # Count the step into the position: forward, along where the nose points.
                self.x += WALK_SPEED * WALK_SCALE * dt * math.cos(math.radians(self.heading))
                self.y += WALK_SPEED * WALK_SCALE * dt * math.sin(math.radians(self.heading))

        if self.state != State.IDLE:
            self.dog.send_speed(vx, 0.0, wz, now)
        self.read_heading(wz, dt)

    @staticmethod
    def call_now(function: Callable) -> Future:
        """Run it on the spot, and hand back a finished Future (what the thread pool would)."""
        done: Future = Future()
        try:
            done.set_result(function())
        except Exception as error:
            done.set_exception(error)
        return done

    # ---------------------------------------------------------------- for the map page
    def snapshot(self) -> dict:
        here = np.array([self.x, self.y])
        report = {
            "state": self.state.name, "job": self.job, "message": self.message,
            "pose": {"x": self.x, "y": self.y, "heading_deg": wrap(self.heading)},
            "trail": [list(p) for p in self.trail],
            "goal": None if self.goal is None or self.job != "goto" else [float(self.goal[0]), float(self.goal[1])],
            "echoes": [{"bearing_deg": e.bearing_deg, "distance_m": e.distance_m, "strength": e.strength,
                        "spread_deg": e.spread_deg} for e in self.echoes],
            "rings": [{"bearing_deg": e.bearing_deg, "distance_m": e.distance_m, "strength": e.strength} for e in self.rings],
            "map": self.map.as_dict(),
            "outline": self.map.outline(here) if self.map.looks else [],
            "problems": self.problems,
            "ears": {"mode": self.ears.geometry.mode, "names": self.ears.geometry.names,
                     "needs_turning": bool(self.ears.geometry.needs_turning)},
        }
        if self.hearing is not None:
            # Each ear's profile against the wall distance it would mean (half the path), thinned for the page.
            keep = slice(None, None, 4)
            report["profiles"] = {
                "wall_m": [round(float(v) / 2, 3) for v in self.hearing.path_m[0][keep]],
                "ears": [[round(float(v), 3) for v in curve[keep]] for curve in self.hearing.reflectivity],
            }
        return report
