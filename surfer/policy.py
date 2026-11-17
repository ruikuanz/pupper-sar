"""What to do next: the rows ahead + the lane Pupper is in -> one of the five moves.

    run     nothing to do (the default: keep running in place)
    left    shuffle one lane left
    right   shuffle one lane right
    jump    a hurdle is about to arrive in this lane
    squat   a beam is about to arrive in this lane

Two different kinds of timing:

  - A lane change should happen as EARLY as possible: it is slow (most of a
    second on the real dog), and being in the right lane early costs nothing.
  - A jump or a squat must happen at the RIGHT moment: the body has to be in
    the air (or down) exactly when the obstacle passes. So the policy waits
    until the obstacle is `lead` seconds away, where the lead is how long the
    move takes from "decide" to "halfway through".

Which lane? Each lane Pupper could be in for the next row (its own, or one
next to it) gets a cost: what stands there (a wall is as good as infinite, a
hurdle or beam costs a trick), plus the shuffle to get there, plus a look at
the row after it. The cheapest wins. With TRICK_COST below SHUFFLE_COST,
Pupper jumps a hurdle in its lane rather than walking around it: all five
moves show up in a game.

No ROS and no clock in this file: decide() is a function of what it is given.
"""

from __future__ import annotations

from dataclasses import dataclass

from perception import BEAM, FREE, HURDLE, WALL, Row

WALL_COST = 100.0
TRICK_COST = 0.3       # jumping or squatting for a row
SHUFFLE_COST = 0.5     # changing one lane
LOOKAHEAD_WEIGHT = 0.6  # how much the row after next counts
TOO_LATE_COST = 50.0   # a lane that can't be reached, or prepared for, in time
Z_ARRIVED = -0.3       # a row this far behind the runner no longer matters


@dataclass
class Timing:
    """How long Pupper's moves take, in seconds (moves.py fills this in)."""

    shuffle_s: float = 0.8    # one lane change, start to finish
    jump_lead_s: float = 0.72  # from deciding to jump until the top of the jump
    squat_lead_s: float = 0.6  # from deciding to squat until the middle of the squat
    margin_s: float = 0.15     # slack on "can this still be done in time?"


def cell_cost(cell: str) -> float:
    return {FREE: 0.0, HURDLE: TRICK_COST, BEAM: TRICK_COST, WALL: WALL_COST}[cell]


def best_lane(rows: list[Row], lane: int, speed: float, timing: Timing) -> int:
    """The lane to be in for the nearest row: this one, or one next to it."""
    nearest = rows[0]
    time_left = nearest.z / speed
    after = rows[1] if len(rows) > 1 else None
    best, best_cost = lane, float("inf")
    for candidate in (lane, lane - 1, lane + 1):  # its own lane first: ties stay put
        if not 0 <= candidate <= 2:
            continue
        cell = nearest.cells[candidate]
        cost = cell_cost(cell)
        if candidate != lane:
            cost += SHUFFLE_COST
            # Halfway through a shuffle the runner counts as in the new lane.
            needed = 0.5 * timing.shuffle_s + timing.margin_s
            if cell in (HURDLE, BEAM):  # ... and then there must be time left for the trick
                needed = timing.shuffle_s + max(timing.jump_lead_s, timing.squat_lead_s)
            if time_left < needed:
                cost += TOO_LATE_COST
        if after is not None:
            cost += LOOKAHEAD_WEIGHT * min(
                cell_cost(after.cells[c]) + SHUFFLE_COST * abs(c - candidate)
                for c in (candidate - 1, candidate, candidate + 1) if 0 <= c <= 2
            )
        if cost < best_cost:
            best, best_cost = candidate, cost
    return best


def decide(rows: list[Row], lane: int, speed: float, timing: Timing = Timing()) -> tuple[str, Row | None]:
    """The move to start now ("run" for none), and the row it is for.

    rows: RowTracker's rows, nearest first. lane: where Pupper is (0, 1, 2).
    speed: track units per second. The caller only asks while Pupper is free
    to start a move, and marks row.handled when it starts a jump or a squat.
    """
    ahead = [r for r in rows if r.z > Z_ARRIVED]
    if not ahead or speed <= 0:
        return "run", None
    nearest = ahead[0]
    target = best_lane(ahead, lane, speed, timing)
    if target != lane:
        return ("left" if target < lane else "right"), nearest

    cell = nearest.cells[lane]
    time_left = nearest.z / speed
    if nearest.handled:
        return "run", None
    if cell == HURDLE and time_left <= timing.jump_lead_s:
        return "jump", nearest
    if cell == BEAM and time_left <= timing.squat_lead_s:
        return "squat", nearest
    return "run", None
