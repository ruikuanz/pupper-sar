"""Pupper's eyes on the real game: one picture of Subway Surfers -> what is ahead.

perception.py reads Lane Hound, a game we drew ourselves, so every obstacle
has its own flat colour. The real game was drawn for people, not for robots:
this file reads it the way a beginner does, with two questions per picture.

  1. Is there a striped barrier ahead? The red-and-white warning boards are
     the one thing in the game that looks the same in every city:

         low and wide, a chevron pointing up      jump over it
         tall and narrow, chevrons pointing down  roll under it
         low and wide, upright stripes (a bar)    the end of a track: go around

     How far away: by how wide the board looks. A board twice as wide is half
     as far (a person follower measures distance the same way, by the height
     of the box around the person).

  2. Is the track in front of me open? The runner is always running on track,
     so the picture just under its feet shows what "open track" looks like
     right now, in this city, in this light. A lane is open as far as the
     picture keeps looking like that. Whatever ends it, and is not a striped
     board, is a train (or a wall, a pillar, a bush: anything to go around).

The picture is always the whole game view, resized to VIEW_W x VIEW_H. The
camera in the game follows the runner most of the way, so the runner is always
near the middle of the picture (a little left of it in the left lane, a little
right in the right lane) and the three lanes are "mine, the one to my left, the
one to my right": offsets -1, 0, +1. The reader has to be told which lane the
runner is in (0, 1, 2); the pilot keeps count.

Everything here was tuned on the browser release of the game (640 x 360).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

VIEW_W, VIEW_H = 640, 360

# --- where things are in the picture (pixels of the 640 x 360 view) -----------
# Measured on a frame with all three tracks in view: the lanes' centre lines
# are straight lines that meet above the runner, at row VANISH_Y.
RUNNER_X = (283, 318, 353)   # the runner's column, in the left, centre and right lane
REF_Y = 300           # the row at which LANE_PITCH is measured
FEET_Y = 330          # the row of the runner's feet
HEAD_Y = 246          # about the top of its head (it bobs): track in its own lane is read from here up
HEAD_CLEAR_Y = 224    # ... and by this row the head is certainly out of the way
VANISH_Y = 52         # the row where the lanes meet
LANE_PITCH = 120.0    # px from one lane's centre to the next, at REF_Y
SCAN_TOP_Y = 104      # do not read lanes above this row: too far, too small

# --- the striped boards ----------------------------------------------------------
RED_HUE_MIN, RED_HUE_MAX = 165, 6   # OpenCV hue wraps at 180: 165..180 and 0..6
RED_MIN_S, RED_MIN_V = 110, 170
BOARD_MIN_W = 14          # px: smaller than this is too far to trust
BOARD_MIN_RED = 40        # px of red in it
BOARD_FILL = 0.22         # at least this much of the box is red
WIDE_BOARD = 1.6          # width / height above this: a low board (to jump)
UPRIGHT_STRIPES = 0.25    # a bar: its columns are all red or all white (see striped())
SURE_WIDTH = 24           # px: a low board narrower than this is too small to tell chevron from bar
# A board's width in px when it is level with the runner (a lane is LANE_PITCH
# wide there), and how fast 1/width shrinks at the game's starting speed
# (1/px per second). From recordings of the browser game. The bar is a much
# wider thing than the chevron board, so the two have their own numbers; "low"
# is a low board still too far away to tell which of the two it is.
WIDTH_AT_RUNNER = {"jump": 52.0, "low": 52.0, "roll": 57.0, "bar": 95.0}
CLOSING_RATE = {"jump": 0.031, "low": 0.031, "roll": 0.042, "bar": 0.034}

# --- open track --------------------------------------------------------------------
TRACK_PATCH_Y = (296, 326)  # rows of track level with the runner's legs ...
TRACK_PATCH_X = (24, 50)    # ... this far to its left and to its right: still its own lane, and
                            # not behind it, where the guard and his dog run at the start
TRACK_MEMORY = 0.2          # how much one picture changes the remembered track colour
TRACK_NEW_AFTER = 8         # pictures in a row of some other colour before it is believed (a train's roof to run on)
TRACK_TOLERANCE_AB = 8.0    # how far a row's colour (Lab a, b) may be from the track's
TRACK_DARKER, TRACK_LIGHTER = 75.0, 80.0   # ... and its lightness (sleepers, rails, shadows)
TRACK_FOLLOW = 0.25         # far track is hazier: the colour looked for follows the rows already read
STRIP = 0.30                # the middle part of a lane that is looked at (x lane width)
OPEN_FILL = 0.65            # a row is open track if this much of the strip looks like track
BLOCK_ROWS = 4              # this many closed rows in a row: something is there


@dataclass
class Board:
    """A striped barrier."""

    kind: str        # "jump" (chevron: jump over), "roll" (roll under), "bar" (stripes: go around), "low" (jump or bar, too far to tell)
    lane: int        # -1, 0, +1: left of me, my lane, right of me
    x: float         # centre column
    y: float         # top row
    w: float
    h: float
    seconds: float   # until it reaches the runner, at the game's starting speed


@dataclass
class Scene:
    boards: list[Board] = field(default_factory=list)
    # For lanes -1, 0, +1: the row where open track ends (smaller = further away),
    # or None if the lane is open as far as we look.
    blocked_y: dict[int, float | None] = field(default_factory=dict)
    track_ok: bool = True   # False: could not tell what open track looks like
    playing: bool = True    # False: this is a menu or the game-over screen, not a run
    lane: int = 1           # the lane the runner was said to be in

    def seconds(self, lane: int) -> float | None:
        """Until open track ends in a lane (starting speed); None if it is open all the way."""
        end = self.blocked_y.get(lane)
        return None if end is None else seconds_from_row(end)

    def board(self, lane: int = 0) -> Board | None:
        """The nearest board in a lane."""
        mine = [b for b in self.boards if b.lane == lane]
        return min(mine, key=lambda b: b.seconds) if mine else None

    def summary(self) -> str:
        boards = " ".join(f"{b.kind}{b.lane:+d}@{b.seconds:.2f}s(w{b.w:.0f})" for b in self.boards)
        lanes = " ".join(f"{k:+d}:{'open' if v is None else int(v)}" for k, v in sorted(self.blocked_y.items()))
        return f"[{boards}] {lanes}"


# ----------------------------------------------------------------------- geometry
def lane_pitch(y: float) -> float:
    """Distance between lane centres at row y: zero at the vanishing row."""
    return LANE_PITCH * (y - VANISH_Y) / (REF_Y - VANISH_Y)


def lane_x(offset: int, y: float, lane: int = 1) -> float:
    """Column of the lane `offset` lanes to the right of the runner, at row y."""
    return RUNNER_X[lane] + offset * lane_pitch(y)


def seconds_from_width(width: float, kind: str) -> float:
    """How long until a board this wide reaches the runner (starting speed)."""
    return max(0.0, (1.0 / width - 1.0 / WIDTH_AT_RUNNER[kind]) / CLOSING_RATE[kind])


def seconds_from_row(y: float) -> float:
    """How long until something standing on the track at row y reaches the runner.

    A bar is a fixed part of a lane wide, so a bar standing at row y would
    look as wide as the lane is there, to scale; and seconds_from_width()
    knows how long a bar of that width has left."""
    width = WIDTH_AT_RUNNER["bar"] * lane_pitch(y) / lane_pitch(FEET_Y)
    return seconds_from_width(max(width, 1.0), "bar")


# ----------------------------------------------------------------------- boards
def red_mask(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (((h >= RED_HUE_MIN) | (h <= RED_HUE_MAX)) & (s > RED_MIN_S) & (v > RED_MIN_V)).astype(np.uint8)


def merge_pieces(boxes: list[list[int]]) -> list[list[int]]:
    """A board's white stripes cut its red into pieces: put pieces that sit side by side, or one above the other, back together."""
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                (x0, y0, x1, y1), (u0, v0, u1, v1) = boxes[i], boxes[j]
                tall, wide = max(y1 - y0, v1 - v0), max(x1 - x0, u1 - u0)
                beside = min(y1, v1) - max(y0, v0) >= 0.5 * min(y1 - y0, v1 - v0) and max(x0, u0) - min(x1, u1) <= 0.7 * tall
                stacked = min(x1, u1) - max(x0, u0) >= 0.5 * min(x1 - x0, u1 - u0) and max(y0, v0) - min(y1, v1) <= 0.5 * wide
                if beside or stacked:
                    boxes[i] = [min(x0, u0), min(y0, v0), max(x1, u1), max(y1, v1)]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def striped(red: np.ndarray) -> float:
    """0 for upright stripes (every column all red or no red), larger for chevrons."""
    columns = red.mean(axis=0)
    return float(np.minimum(columns, 1.0 - columns).mean())


def find_boards(hsv: np.ndarray, lane: int = 1) -> list[Board]:
    red = red_mask(hsv)
    red[:VANISH_Y + 40] = 0     # the sky can be red too
    red[FEET_Y:] = 0
    joined = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, _labels, stats, _centres = cv2.connectedComponentsWithStats(joined)
    pieces = [[x, y, x + w, y + h] for x, y, w, h, area in stats[1:count] if area >= 12 and h >= 5]
    boards = []
    for x0, y0, x1, y1 in merge_pieces(pieces):
        w, h = x1 - x0, y1 - y0
        if w < BOARD_MIN_W or h < 6 or w > 130:
            continue
        inside = red[y0:y1, x0:x1]
        reds = int(inside.sum())
        if reds < BOARD_MIN_RED or reds < BOARD_FILL * w * h:
            continue
        if not 0.6 <= w / h <= 4.5:
            continue  # train sides are long streaks, boards are not
        if w / h < WIDE_BOARD:
            kind = "roll"
        elif w < SURE_WIDTH:
            kind = "low"
        else:
            kind = "bar" if striped(inside) < UPRIGHT_STRIPES else "jump"
        centre = x0 + w / 2
        # Which lane: compare with the lane centres at the depth of the board,
        # found from its width (a board is a fixed fraction of a lane wide).
        pitch = lane_pitch(FEET_Y) * w / WIDTH_AT_RUNNER[kind]
        offset = int(round((centre - RUNNER_X[lane]) / max(pitch, 1.0)))
        if abs(offset) > 2:
            continue
        boards.append(Board(kind, offset, centre, float(y0), float(w), float(h), seconds_from_width(w, kind)))
    return boards


# ----------------------------------------------------------------------- open track
def track_colour(lab: np.ndarray, lane: int) -> tuple[np.ndarray, bool]:
    """What open track looks like right now: the colour (L, a, b) beside the runner's legs."""
    x = RUNNER_X[lane]
    (y0, y1), (near, far) = TRACK_PATCH_Y, TRACK_PATCH_X
    patch = np.concatenate([lab[y0:y1, x - far:x - near].reshape(-1, 3), lab[y0:y1, x + near:x + far].reshape(-1, 3)])
    colour = np.median(patch, axis=0)
    spread = float(np.abs(patch[:, 1:] - colour[1:]).mean())
    return colour, spread < 0.75 * TRACK_TOLERANCE_AB


def open_until(lab: np.ndarray, colour: np.ndarray, offset: int, lane: int) -> float | None:
    """Scan one lane from the runner outwards; the row where open track ends."""
    start = HEAD_Y if offset == 0 else FEET_Y   # my own lane: the runner hides the rows below its head
    light, a, b = (float(c) for c in colour)
    closed = 0
    seen_open = offset != 0   # my own lane: skip rows until the runner's head is out of the way
    for y in range(start, SCAN_TOP_Y, -1):
        half = max(2.0, 0.5 * STRIP * lane_pitch(y))
        centre = lane_x(offset, y, lane)
        x0, x1 = int(centre - half), int(centre + half) + 1
        if x0 < 0 or x1 > VIEW_W:
            closed += 1   # off the picture: can't see it, don't trust it
        else:
            strip = lab[y, x0:x1]
            like = ((np.abs(strip[:, 1] - a) < TRACK_TOLERANCE_AB) & (np.abs(strip[:, 2] - b) < TRACK_TOLERANCE_AB)
                    & (strip[:, 0] > light - TRACK_DARKER) & (strip[:, 0] < light + TRACK_LIGHTER))
            if like.mean() < OPEN_FILL:
                closed += 1 if seen_open or y < HEAD_CLEAR_Y else 0
            else:
                closed, seen_open = 0, True
                # Follow the slow change of colour with distance (haze, a tunnel's shade).
                a += TRACK_FOLLOW * (float(strip[like, 1].mean()) - a)
                b += TRACK_FOLLOW * (float(strip[like, 2].mean()) - b)
        if closed >= BLOCK_ROWS:
            return float(y + BLOCK_ROWS - 1)
    return None


# ----------------------------------------------------------------------- which screen is this?
PLAY_BUTTON = (326, 314, 417, 341)    # x0, y0, x1, y1 of the green PLAY button on the score screen
SAVE_ME_BUTTON = (267, 156, 336, 189) # the green "Free!" button of the "Save me!" box (never clicked)


def green_part(hsv: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = box
    part = hsv[y0:y1, x0:x1]
    return float(((part[..., 0] > 40) & (part[..., 0] < 75) & (part[..., 1] > 120) & (part[..., 2] > 120)).mean())


def in_a_run(hsv: np.ndarray) -> bool:
    """During a run the pause button is top left and the score, white on black, top right."""
    pause = hsv[8:36, 8:36].reshape(-1, 3)
    blue = ((pause[:, 0] > 95) & (pause[:, 0] < 115) & (pause[:, 1] > 120) & (pause[:, 2] > 150)).mean()
    score = hsv[9:25, 575:632, 2]
    return bool(blue > 0.5 and (score < 80).mean() > 0.45)


def screen(view_bgr: np.ndarray) -> str:
    """"run", "save_me" (the box that offers to continue), "score" (game over, with a PLAY button) or "menu"."""
    if view_bgr.shape[1] != VIEW_W or view_bgr.shape[0] != VIEW_H:
        view_bgr = cv2.resize(view_bgr, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2HSV)
    if in_a_run(hsv):
        return "run"
    if green_part(hsv, PLAY_BUTTON) > 0.5:
        return "score"
    if green_part(hsv, SAVE_ME_BUTTON) > 0.5:
        return "save_me"
    return "menu"


# ----------------------------------------------------------------------- one picture
class Reader:
    """Reads pictures one after the other, remembering what the track looks like.

    For a moment now and then the track beside the runner is hidden (a coin, a
    flash, the runner itself in the middle of a move). Then the colour from the
    pictures before is used."""

    def __init__(self) -> None:
        self.colour: np.ndarray | None = None
        self.strange = 0   # pictures in a row in which the track beside the runner was some other colour

    def read(self, view_bgr: np.ndarray, lane: int = 1) -> Scene:
        """lane: where the runner is (0 left, 1 centre, 2 right)."""
        if view_bgr.shape[1] != VIEW_W or view_bgr.shape[0] != VIEW_H:
            view_bgr = cv2.resize(view_bgr, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        colour, ok = track_colour(lab, lane)
        if ok and self.colour is not None and float(np.abs(colour[1:] - self.colour[1:]).max()) > TRACK_TOLERANCE_AB:
            # Not the colour it was: a coin in the way, or really a new surface?
            self.strange += 1
            if self.strange >= TRACK_NEW_AFTER:
                self.colour, self.strange = colour, 0
        elif ok:
            self.strange = 0
            self.colour = colour if self.colour is None else self.colour + TRACK_MEMORY * (colour - self.colour)
        scene = Scene(boards=find_boards(hsv, lane), track_ok=self.colour is not None, playing=in_a_run(hsv), lane=lane)
        for offset in (-1, 0, 1):
            scene.blocked_y[offset] = open_until(lab, self.colour, offset, lane) if scene.track_ok else None
        return scene


def read(view_bgr: np.ndarray, lane: int = 1) -> Scene:
    """One picture on its own (no memory)."""
    return Reader().read(view_bgr, lane)


def draw_debug(view_bgr: np.ndarray, scene: Scene, note: str = "") -> np.ndarray:
    """The picture with what was read from it drawn on top."""
    if view_bgr.shape[1] != VIEW_W or view_bgr.shape[0] != VIEW_H:
        view_bgr = cv2.resize(view_bgr, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)
    out = view_bgr.copy()
    for offset in (-1, 0, 1):
        end = scene.blocked_y.get(offset)
        top = int(end) if end is not None else SCAN_TOP_Y
        start = HEAD_Y if offset == 0 else FEET_Y
        colour = (60, 255, 60) if end is None else (0, 200, 255)
        cv2.line(out, (int(lane_x(offset, start, scene.lane)), start), (int(lane_x(offset, top, scene.lane)), top), colour, 1)
        if end is not None:
            half = int(0.5 * STRIP * lane_pitch(top)) + 2
            x = int(lane_x(offset, top, scene.lane))
            cv2.line(out, (x - half, top), (x + half, top), (0, 0, 255), 2)
    for b in scene.boards:
        colour = {"jump": (0, 255, 255), "low": (0, 160, 255), "roll": (255, 255, 0), "bar": (0, 0, 255)}[b.kind]
        cv2.rectangle(out, (int(b.x - b.w / 2), int(b.y)), (int(b.x + b.w / 2), int(b.y + b.h)), colour, 1)
        cv2.putText(out, f"{b.kind} {b.seconds:.2f}s", (int(b.x - b.w / 2), int(b.y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    if note:
        cv2.putText(out, note, (8, VIEW_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
