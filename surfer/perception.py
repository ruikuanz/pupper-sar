"""What Pupper sees: one picture of the game -> the obstacle rows ahead.

The input is the *game view*: the inside of the green frame, upright and
undistorted (screen_finder.py gets it out of a camera picture; in stream mode
the browser sends it directly). The output is a list of rows, each with a
distance and what stands in each of the three lanes.

Two ideas do the work:

  1. Color is the class. The game draws every obstacle kind in one saturated
     color and nothing else that is both bright and saturated, so a hue test
     is the detector. No model to train, and it runs in a millisecond.

  2. The bottom edge is a distance sensor. Everything in the game stands on
     the same floor, and the floor recedes upward in the picture, so the row
     where an obstacle touches the ground says how far away it is. The game's
     projection (game/js/render.js) puts a point z units down the track at

         s = Z0 / (z + Z0)        y = HORIZON + (PLAYER - HORIZON) * s

     and z_from_y below is that formula turned around.

How it reads a frame (detect): for each lane it walks up the picture along the
lane's centre line, from the runner toward the horizon, and looks at a short
strip of pixels across the lane at every image row. A run of rows whose strip
is mostly one obstacle color is one obstacle; the bottom of the run is where
it touches the ground.

RowTracker then follows rows from frame to frame. The runner on screen hides
whatever is right behind it, so the last half second before an obstacle
arrives is not visible at all: the tracker carries each row forward at the
game's speed, which is also how the policy gets a steady time-to-impact.

No ROS in this file: python3 tests/test.py runs it on pictures it draws itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CONFIG = json.loads((Path(__file__).resolve().parent.parent / "game" / "config.json").read_text())

FREE, HURDLE, BEAM, WALL = "free", "hurdle", "beam", "wall"
KINDS = (HURDLE, BEAM, WALL)

# The size perception works at. A camera picture of the screen is warped to this.
VIEW_W = CONFIG["view"]["stream_width"]
VIEW_H = CONFIG["view"]["stream_height"]

# The game's projection (see the docstring).
HORIZON = CONFIG["projection"]["horizon"]
PLAYER = CONFIG["projection"]["player"]
Z0 = CONFIG["projection"]["z0"]
LANE_WIDTH = CONFIG["projection"]["lane_width"]

# --- the color test ----------------------------------------------------------
# A pixel belongs to an obstacle kind if its hue is within HUE_TOLERANCE of that
# kind's color and it is saturated and bright enough. OpenCV hues run 0..179.
# Through a camera, colors wash out: if obstacles are missed, lower MIN_SATURATION
# and MIN_VALUE; if the background lights up in the debug view, raise them.
HUE_TOLERANCE = 13
MIN_SATURATION = 90
MIN_VALUE = 90

# --- reading a lane ----------------------------------------------------------
# The strip is narrow and must be mostly filled, for a reason: a tall wall in the
# middle lane reaches up the picture to where the side lanes have converged
# toward it, and its top corners poke into their strips. With these numbers
# (and walls no taller than 1.3 lane widths, config.json) it never fills one.
STRIP_HALF_WIDTH = 0.25   # of a lane: the strip of pixels checked across the lane centre
STRIP_FILL = 0.6          # this much of the strip must be the color
MIN_RUN_ROWS = 2          # image rows an obstacle must span to count
SAME_OBSTACLE = 0.8       # lane widths: see detect (a beam's bar is 0.62 above its stripe)
Z_NEAREST = -1.0          # don't look for obstacles that are already behind the runner
Z_FARTHEST = 70.0         # beyond this an obstacle is a couple of pixels: ignore

# --- following rows ----------------------------------------------------------
ROW_MERGE = 0.22          # detections within this fraction of z (or ROW_MERGE_MIN) are one row
ROW_MERGE_MIN = 2.5       # track units
ROW_FORGET_S = 0.6        # a row that should be visible but isn't seen this long is dropped
Z_HIDDEN = 6.0            # nearer than this the runner may hide a row: keep it on prediction
Z_PASSED = -1.5           # a row this far behind the runner is done
SPEED_WINDOW = 12         # measurements kept for the speed estimate
SPEED_MIN_SAMPLES = 3     # until there are this many, use the game's nominal speed
SPEED_BASELINE_S = 0.4    # time between the two looks one measurement is made from
SPEED_Z_MIN, SPEED_Z_MAX = 5.0, 22.0  # ... while the row is in this range of distances


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    n = int(color.lstrip("#"), 16)
    return (n & 255, (n >> 8) & 255, (n >> 16) & 255)


def hue_of(color: str) -> int:
    """The OpenCV hue (0..179) of a "#RRGGBB" color."""
    pixel = np.uint8([[hex_to_bgr(color)]])
    return int(cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0, 0])


HUES = {kind: hue_of(CONFIG["palette"][kind]) for kind in KINDS}
BORDER_HUE = hue_of(CONFIG["palette"]["border_pupper"])


def hue_mask(hsv: np.ndarray, hue: int, tolerance: int = HUE_TOLERANCE,
             min_saturation: int = MIN_SATURATION, min_value: int = MIN_VALUE) -> np.ndarray:
    """True where a pixel is this hue (hue wraps around at 180), saturated and bright."""
    h = hsv[..., 0].astype(np.int16)
    distance = np.abs(h - hue)
    distance = np.minimum(distance, 180 - distance)
    return (distance <= tolerance) & (hsv[..., 1] >= min_saturation) & (hsv[..., 2] >= min_value)


# --- the projection, both ways -----------------------------------------------
def scale_at(z: float) -> float:
    """How big things are drawn z units down the track (1 at the runner)."""
    return Z0 / (z + Z0)


def y_of(z: float) -> float:
    """Where the ground at distance z is, as a fraction of the view's height."""
    return HORIZON + (PLAYER - HORIZON) * scale_at(z)


def z_from_y(y: float) -> float:
    """The distance of the ground point drawn at height fraction y: y_of, turned around."""
    s = (y - HORIZON) / (PLAYER - HORIZON)
    return float("inf") if s <= 1e-6 else Z0 / s - Z0


BEAM_BOTTOM = CONFIG["obstacles"]["beam"]["bottom"]  # lane widths the beam's bar hangs above the ground


def z_from_bar(y: float, aspect: float) -> float:
    """The distance of a beam whose bar's bottom edge is drawn at height fraction y.

    The bar is BEAM_BOTTOM lane widths above the ground, and a lane width at
    scale s is LANE_WIDTH * s of the view's width (aspect = view width / height):
        y = HORIZON + (PLAYER - HORIZON) * s - BEAM_BOTTOM * LANE_WIDTH * aspect * s
    """
    s = (y - HORIZON) / ((PLAYER - HORIZON) - BEAM_BOTTOM * LANE_WIDTH * aspect)
    return float("inf") if s <= 1e-6 else Z0 / s - Z0


def lane_x(lane: int, z: float) -> float:
    """The centre of a lane (0 left, 1 centre, 2 right) at distance z, as a fraction of the width."""
    return 0.5 + (lane - 1) * LANE_WIDTH * scale_at(z)


@dataclass
class Hazard:
    """One obstacle in one frame."""

    lane: int       # 0 left, 1 centre, 2 right
    kind: str       # hurdle, beam or wall
    z: float        # track units ahead of the runner, from its bottom edge
    y_bottom: int   # the image row of its bottom edge
    y_top: int
    exact: bool = True  # False if something nearer hides its bottom edge
    parts: int = 1      # runs of its color it is made of (a beam seen whole: stripe + bar)


def _strips(width: int, height: int):
    """For every lane and image row: the columns of the strip across the lane centre."""
    ys = np.arange(height)
    s = ((ys + 0.5) / height - HORIZON) / (PLAYER - HORIZON)
    table = []
    for lane in range(3):
        centre = (0.5 + (lane - 1) * LANE_WIDTH * s) * width
        half = np.maximum(STRIP_HALF_WIDTH * LANE_WIDTH * s * width, 1.0)
        x0 = np.clip(np.floor(centre - half), 0, width - 1).astype(int)
        x1 = np.clip(np.ceil(centre + half), 1, width).astype(int)
        table.append((x0, np.maximum(x1, x0 + 1)))
    return table


_STRIP_CACHE: dict[tuple[int, int], list] = {}


def detect(view_bgr: np.ndarray) -> list[Hazard]:
    """Every obstacle in one picture of the game view, nearest first."""
    height, width = view_bgr.shape[:2]
    hsv = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2HSV)
    strips = _STRIP_CACHE.setdefault((width, height), _strips(width, height))
    # Only rows of the picture where the ground is: from Z_FARTHEST down to Z_NEAREST.
    y_first = max(0, int(np.ceil(y_of(Z_FARTHEST) * height)))
    y_last = min(height - 1, int(y_of(Z_NEAREST) * height))
    rows = np.arange(height)

    # labels[lane][y]: which kind (index + 1) fills the strip at that row, or 0.
    labels = np.zeros((3, height), dtype=np.int8)
    for k, kind in enumerate(KINDS):
        mask = hue_mask(hsv, HUES[kind])
        sums = np.concatenate([np.zeros((height, 1), np.int32), np.cumsum(mask, axis=1, dtype=np.int32)], axis=1)
        for lane, (x0, x1) in enumerate(strips):
            filled = (sums[rows, x1] - sums[rows, x0]) >= STRIP_FILL * (x1 - x0)
            labels[lane][filled] = k + 1

    hazards = []
    for lane in range(3):
        y = y_last
        touching = False  # is the run being read sitting right on top of a nearer one?
        last = None       # the last obstacle found in this lane
        while y >= y_first:
            label = labels[lane][y]
            if label == 0:
                touching = False
                y -= 1
                continue
            bottom = y
            while y >= y_first and labels[lane][y] == label:
                y -= 1
            top = y + 1
            kind = KINDS[label - 1]
            if bottom - top + 1 >= MIN_RUN_ROWS:
                # A beam is a stripe on the ground with its bar floating above it: two
                # runs of one color, a little apart. A run that starts less than
                # SAME_OBSTACLE lane widths above the bottom of the last one, in the
                # same color, is the upper part of that obstacle, not another one.
                unit = LANE_WIDTH * width * (((last.y_bottom + 1) / height - HORIZON) / (PLAYER - HORIZON)) if last else 0.0
                if last and last.kind == kind and last.y_bottom - bottom <= SAME_OBSTACLE * unit:
                    last.y_top = top
                    last.parts += 1
                else:
                    # Touching a nearer obstacle from above: its real bottom edge is hidden.
                    last = Hazard(lane, kind, z_from_y((bottom + 1) / height), bottom, top, exact=not touching)
                    hazards.append(last)
            touching = True

    # Far away, a beam's stripe on the ground is too thin to see, and what is
    # left is its bar: a bottom edge that is NOT on the floor. This is the case
    # where "bottom edge = distance" lies, and the fix is to know
    # the object: the bar hangs BEAM_BOTTOM lane widths above the ground.
    for hazard in hazards:
        if hazard.kind == BEAM and hazard.parts == 1:
            hazard.z = z_from_bar((hazard.y_bottom + 1) / height, width / height)
    hazards.sort(key=lambda h: h.z)
    return hazards


@dataclass
class Row:
    """One row of obstacles across the track, followed from frame to frame."""

    z: float
    votes: list[dict] = field(default_factory=lambda: [{}, {}, {}])  # per lane: kind -> times seen
    seen_at: float = 0.0
    handled: bool = False  # the policy already jumped or squatted for this row
    anchor: tuple[float, float] | None = None  # (z, time) of an earlier clear look, for the speed

    @property
    def cells(self) -> list[str]:
        """What stands in each lane: the kind seen most often there, else free."""
        return [max(v, key=v.get) if v else FREE for v in self.votes]


class RowTracker:
    """Follows rows over time, and measures how fast the track is coming."""

    def __init__(self, nominal_speed: float = CONFIG["presets"]["pupper"]["speed"]):
        self.nominal_speed = nominal_speed
        self.rows: list[Row] = []
        self.last_time: float | None = None
        self.speed_samples: list[float] = []

    def reset(self, nominal_speed: float | None = None) -> None:
        if nominal_speed:
            self.nominal_speed = nominal_speed
        self.rows, self.last_time, self.speed_samples = [], None, []

    @property
    def speed(self) -> float:
        """Track units per second. Measured from how rows approach; until there are
        enough measurements (and never far from it), the game's nominal speed."""
        if len(self.speed_samples) < SPEED_MIN_SAMPLES:
            return self.nominal_speed
        measured = float(np.median(self.speed_samples))
        return min(max(measured, 0.6 * self.nominal_speed), 1.6 * self.nominal_speed)

    def measure_speed(self, row: Row, z: float, now: float) -> None:
        """How far a row came in SPEED_BASELINE_S, from two clear looks at it. Only
        while it is near enough that a pixel is a small distance (SPEED_Z_MAX)."""
        if z > SPEED_Z_MAX or z < SPEED_Z_MIN or row.anchor is None or row.anchor[0] > SPEED_Z_MAX:
            if row.anchor is None or row.anchor[0] > SPEED_Z_MAX:
                row.anchor = (z, now)
            return
        z_then, then = row.anchor
        if now - then >= SPEED_BASELINE_S:
            self.speed_samples = (self.speed_samples + [(z_then - z) / (now - then)])[-SPEED_WINDOW:]
            row.anchor = (z, now)

    def predict(self, now: float) -> None:
        """Carry every row forward to `now` at the current speed."""
        if self.last_time is not None:
            step = self.speed * max(0.0, now - self.last_time)
            for row in self.rows:
                row.z -= step
        self.last_time = now
        self.rows = [r for r in self.rows if r.z > Z_PASSED]

    def update(self, hazards: list[Hazard], now: float, runner_lanes: tuple[int, ...] = ()) -> list[Row]:
        """Take in one frame's obstacles. Returns the rows, nearest first.

        runner_lanes: the lane the runner is in (both lanes, during a shuffle).
        Close up, the runner stands in front of whatever is in its lane and
        hides the bottom edge, so those obstacles are not trusted for distance.
        """
        self.predict(now)
        for hazard in hazards:
            if hazard.lane in runner_lanes and hazard.z < Z_HIDDEN:
                hazard.exact = False

        # Group this frame's obstacles into rows by distance.
        groups: list[list[Hazard]] = []
        for hazard in hazards:
            near = groups and abs(hazard.z - groups[-1][0].z) <= max(ROW_MERGE_MIN, ROW_MERGE * hazard.z)
            if near:
                groups[-1].append(hazard)
            else:
                groups.append([hazard])

        for group in groups:
            exact = [h.z for h in group if h.exact]
            z = float(np.median(exact)) if exact else float(np.median([h.z for h in group]))
            match = min(self.rows, key=lambda r: abs(r.z - z), default=None)
            if match is not None and abs(match.z - z) <= max(ROW_MERGE_MIN, ROW_MERGE * z):
                if exact:
                    self.measure_speed(match, z, now)
                    match.z += 0.5 * (z - match.z)
                row = match
            elif exact:
                row = Row(z=z, anchor=(z, now))
                self.rows.append(row)
            else:
                continue  # a first sighting with a hidden bottom edge: wait for a better look
            row.seen_at = now
            for hazard in group:
                row.votes[hazard.lane][hazard.kind] = row.votes[hazard.lane].get(hazard.kind, 0) + 1

        # A row that should be in plain view and has not been seen for a while was
        # a false alarm (or the game restarted). Near the runner, a row is kept on
        # prediction: the runner hides it.
        self.rows = [r for r in self.rows if r.z < Z_HIDDEN or now - r.seen_at <= ROW_FORGET_S]
        self.rows.sort(key=lambda r: r.z)
        return self.rows


def draw_debug(view_bgr: np.ndarray, hazards: list[Hazard], rows: list[Row], lane: int, note: str = "") -> np.ndarray:
    """The game view with what was read from it drawn on top (for the web page)."""
    out = view_bgr.copy()
    height, width = out.shape[:2]
    for hazard in hazards:
        x = int(lane_x(hazard.lane, hazard.z) * width)
        half = max(2, int(0.5 * LANE_WIDTH * scale_at(hazard.z) * width))
        cv2.rectangle(out, (x - half, hazard.y_top), (x + half, hazard.y_bottom), (255, 255, 255), 1)
    for row in rows:
        if row.z < Z_NEAREST:
            continue
        y = int(y_of(row.z) * height)
        cv2.line(out, (0, y), (width - 1, y), (255, 255, 255), 1)
        label = " ".join({FREE: ".", HURDLE: "J", BEAM: "S", WALL: "X"}[c] for c in row.cells)
        cv2.putText(out, f"{label}  {row.z:4.1f}", (4, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    x = int(lane_x(lane, 0.0) * width)
    cv2.circle(out, (x, int(PLAYER * height) + 8), 4, (255, 255, 255), -1)
    if note:
        cv2.putText(out, note, (4, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return out
