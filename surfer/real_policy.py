"""Which move for the real game: jump, squat, left, right, or keep running.

policy.py plans a few rows ahead because Lane Hound shows the whole course from
far away. The real game does not: things get big only in the last second. So
this pilot reacts, like a person learning the game:

    a low board close ahead      jump
    a tall board close ahead     squat (the game's roll)
    my lane ends in a train      step to the side lane that is open for longer

It keeps count of which lane the runner is in (0 left, 1 centre, 2 right) from
the moves it has made, and it waits out its own moves: while the runner is in
the air the game's camera rises with it, and the picture cannot be read.

All times are "seconds until it reaches the runner". The picture gives them at
the game's starting speed (real_perception.py); the game gets faster as a run
goes on, so they are divided by `speed`, which is measured from how fast the
boards grow.
"""

from __future__ import annotations

import real_perception as rp

# --- when to move (seconds before the thing reaches the runner) ----------------
JUMP_LEAD_S = 0.30
ROLL_LEAD_S = 0.30
DODGE_LEAD_S = 1.00      # leave a lane that ends in a train this long before
# --- how long the runner in the game is busy after a key ------------------------
GAME_JUMP_S = 0.70
GAME_ROLL_S = 0.70
GAME_STEP_S = 0.28
# --- going around trains -----------------------------------------------------------
SIDE_MIN_S = 0.45        # a side lane must be open at least this long to step into it
BETTER_BY_S = 0.25       # ... and for this much longer than my own lane
OPEN_S = 9.0             # what "open as far as I can see" counts as
START_GRACE_S = 1.0      # a run starts with a flash and the guard in the picture: don't read those
# --- the game's speed ----------------------------------------------------------------
SPEED_MIN, SPEED_MAX = 1.0, 2.5
SPEED_BLEND = 0.3        # how much one new measurement counts
SPEED_BASELINE_S = 0.12  # measure a board's growth over at least this long


class Pilot:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.lane = 1
        self.busy_until = 0.0
        self.speed = 1.0          # the game's speed, relative to its starting speed
        self.anchor = None        # (time, 1/width, kind) of the board being watched, for the speed
        self.playing_since = None # when this run began

    # ------------------------------------------------------------------ speed
    def watch(self, board: rp.Board | None, now: float) -> None:
        """Measure the game's speed from how fast the board ahead grows."""
        if board is None or board.w > 0.9 * rp.WIDTH_AT_RUNNER[board.kind]:
            self.anchor = None
            return
        if board.kind == "low":
            return
        inverse = 1.0 / board.w
        if self.anchor is None or self.anchor[2] != board.kind or inverse > self.anchor[1]:
            self.anchor = (now, inverse, board.kind)
            return
        then, was, kind = self.anchor
        if now - then >= SPEED_BASELINE_S and was - inverse > 0.004:
            measured = (was - inverse) / (now - then) / rp.CLOSING_RATE[kind]
            if 0.6 <= measured <= 3.0:
                blended = self.speed + SPEED_BLEND * (measured - self.speed)
                self.speed = min(SPEED_MAX, max(SPEED_MIN, blended))
            self.anchor = (now, inverse, kind)

    # ------------------------------------------------------------------ one lane
    def ahead(self, scene: rp.Scene, lane: int) -> tuple[float, str | None]:
        """(seconds until something reaches the runner in this lane, what it is).

        What: "jump" or "roll" (a board to do that trick at), "wall" (go
        around: a train, or the striped bar at the end of a track), or None
        with OPEN_S when the lane is open as far as can be seen."""
        seconds, what = OPEN_S, None
        end = scene.seconds(lane)
        if end is not None:
            seconds, what = end, "wall"
        board = scene.board(lane)
        if board is not None and board.seconds <= seconds + 0.35:
            # The board is what ends the open track (or stands before it).
            what = {"jump": "jump", "low": "jump", "roll": "roll", "bar": "wall"}[board.kind]
            seconds = min(seconds, board.seconds)
        return seconds / self.speed, what

    # ------------------------------------------------------------------ one picture
    def step(self, scene: rp.Scene, now: float) -> str | None:
        """One picture in, one move out (or None: keep running)."""
        if not scene.playing:
            self.playing_since = None
            return None
        if self.playing_since is None:
            self.playing_since = now
        if now - self.playing_since < START_GRACE_S or now < self.busy_until:
            self.anchor = None
            return None
        self.watch(scene.board(0), now)

        seconds, what = self.ahead(scene, 0)
        if what == "jump" and seconds <= JUMP_LEAD_S:
            self.busy_until = now + GAME_JUMP_S
            return "jump"
        if what == "roll" and seconds <= ROLL_LEAD_S:
            self.busy_until = now + GAME_ROLL_S
            return "squat"
        if what == "wall" and seconds <= DODGE_LEAD_S and scene.track_ok:
            best, best_s = None, seconds + BETTER_BY_S
            for side, move in ((-1, "left"), (1, "right")):
                if not 0 <= self.lane + side <= 2:
                    continue
                side_s, side_what = self.ahead(scene, side)
                if side_what in ("jump", "roll"):
                    side_s += 0.5   # a board can be dealt with: better than a train at the same distance
                if side_s >= SIDE_MIN_S and side_s > best_s:
                    best, best_s = move, side_s
            if best:
                self.lane += -1 if best == "left" else 1
                self.busy_until = now + GAME_STEP_S
                return best
        return None
