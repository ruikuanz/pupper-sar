"""From what the ears heard to where the walls are: triangulation.

The one big idea
----------------
A flat wall is a mirror for sound. Its echo reaches the ears exactly as if a
second speaker stood behind the wall, at the mirror image of the real one: the
*image source*. Find the image source and you have the wall: it is the plane
halfway between the speaker and its image, face-on to the line joining them.

Each ear's profile (chirp.py) says how far the sound travelled to reach *that
ear*: the image source is somewhere on a circle around that ear with that
radius. Two ears, two circles; they cross at the image source. That crossing
is the triangulation.

    two ears     Two circles cross at two points, mirror images of each other
                 in the line through the ears. Pupper knows the wall's distance
                 and its angle off that line, but not which side: with an ear
                 on each flank, not "ahead" from "behind". Like an owl turning
                 its head, Pupper turns and listens again: the real wall stays
                 put in the room, the phantom jumps (room_map.py sorts them).
    four ears    Three or more circles (ears not in one line) cross at one
                 point only. One ping, no turning. They also reject the
                 ceiling: its echo reaches all four at the same moment, which
                 no wall standing on the floor can do.

In practice we don't pick peaks and pair them up between ears (which peak in
the left ear belongs with which in the right?). We ask the question the other
way round, for every place an image source could be (find_echoes): "if it were
here, when would each ear have heard it, and did it?" A place scores high only
if *every* ear heard something at its moment. Painting each ear's profile onto
the floor plan gives rings around that ear; the score is high where the rings
of all ears cross.

Posts, chair legs and people answer too. They are not mirrors, but the same
arithmetic puts a small face-on wall where they stand, which is the right
thing to not walk into.

What echolocation cannot see: a wall only answers from the one spot where it
is face-on to Pupper (the foot of the perpendicular). A wall running away at
an angle sends its echo off somewhere else. So each echo is one face-on patch
of wall, and "no echo that way" does not mean "nothing there".

Numpy only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

import chirp

HERE = Path(__file__).resolve().parent
EARS_FILE = HERE / "ears.yaml"
CALIBRATION_FILE = HERE / "calibration.json"   # written by mic_check.py --wall

# --- where to look ---------------------------------------------------------------
MIN_RANGE_M = 0.35      # nearer walls hide under the direct sound and its bounce off the floor
MAX_RANGE_M = 3.5       # further ones drown in the room's ringing
RANGE_STEP_M = 0.01
BEARING_STEP_DEG = 2.0

# --- what counts as an echo ------------------------------------------------------
# Profiles are scaled so that a perfect mirror scores 1 at any distance (see
# reflectivity()). Painted walls and cardboard score 0.4-0.9, a chair leg 0.2.
MIN_REFLECTIVITY = 0.22  # weaker than this is not an echo
MIN_ABOVE_FLOOR = 4.0    # ... and it must stand this many times above the typical score
MAX_PEAKS = 24           # per look, before the phantoms are thrown out
SAME_ECHO_M = 0.15       # a peak owns this much range either side of it ...
SAME_ECHO_DEG = 16.0     # ... and at least this much bearing, more if its ridge runs on (RIDGE)
RIDGE = 0.5              # a ridge belongs to its peak while it stays above this share of it
BLUR_M = 0.03            # ruler error we forgive: profiles are smeared by this much path
CEILING_CUT_M = 0.12     # silence this much path either side of the ceiling's echo

# --- phantoms --------------------------------------------------------------------
# 1. Rings. Our ears sit close together, so a strong echo leaves a faint ring
#    of itself at the same distance in other directions. A peak at the same
#    distance as a much stronger one is taken for that ring, not another wall.
#    (Sometimes it IS another wall, a weaker one that happens to be as far
#    away. So rings are kept off the map but Pupper still will not walk into
#    one: navigator.py.)
RING_M = 0.15
RING_SHARE = 0.6         # "much stronger": the peak is under this share of the strong one
# 2. Sound that bounced off two walls looks like a wall *behind* the first one.
# Nothing behind a solid wall can answer, so: an echo behind a nearer, strong
# echo's wall is thrown out. (A weak echo is a chair leg, not a wall: sound
# goes around it, so it hides nothing.)
SOLID_STRENGTH = 0.35
BEHIND_M = 0.12          # this far behind a wall's plane still counts as on it (corners are)


@dataclass
class Geometry:
    """Where the speaker and the ears are on the body (ears.yaml)."""

    mode: str
    names: list[str]
    speaker: np.ndarray      # (3,)  x forward, y left, z up from the floor
    mics: np.ndarray         # (n, 3)
    direct_m: np.ndarray     # (n,)  path of the direct sound to each ear
    ceiling_m: float | None

    @property
    def needs_turning(self) -> bool:
        """True if the ears are in one line: phantoms on the far side, so turn and listen again."""
        flat = self.mics[:, :2] - self.mics[:, :2].mean(axis=0)
        return len(self.mics) < 3 or np.linalg.matrix_rank(flat, tol=0.01) < 2


def load_geometry(mode: str, path: Path = EARS_FILE, calibration: Path | None = CALIBRATION_FILE) -> Geometry:
    """Read ears.yaml for one mode ("two" or "four").

    The direct sound's path to each ear is the ruler distance speaker -> ear,
    unless mic_check.py --wall measured a better one (the sound goes around
    the body, not through it): then calibration.json has it.
    """
    config = yaml.safe_load(Path(path).read_text())
    if mode not in config["modes"]:
        raise ValueError(f"{path} has no mode '{mode}' (it has: {', '.join(config['modes'])})")
    rows = config["modes"][mode]["mics"]
    if len(rows) < 2:
        raise ValueError("echolocation needs at least two ears")
    speaker = np.array(config["speaker"], float)
    mics = np.array([row[1:4] for row in rows], float)
    direct = np.linalg.norm(mics - speaker, axis=1)
    if calibration and Path(calibration).exists():
        saved = json.loads(Path(calibration).read_text()).get(mode, {})
        if len(saved.get("direct_m", [])) == len(mics):
            direct = np.array(saved["direct_m"], float)
    return Geometry(mode, [str(row[0]) for row in rows], speaker, mics, direct, config.get("ceiling_m"))


@dataclass
class Echo:
    """One face-on patch of wall, seen from the middle of the body.

    Bearings are in the frame of the headings given to Ears.locate: give it
    headings in the room and bearings are in the room; give it 0 and they are
    relative to the nose (0 dead ahead, +90 to the left, 180 behind).
    """

    bearing_deg: float    # which way the wall is (the direction straight at it)
    distance_m: float     # from the middle of the body to the wall, straight at it
    strength: float       # 1 = a perfect mirror
    spread_deg: float     # how unsure the bearing is (half the width of the ridge)

    @property
    def toward(self) -> np.ndarray:
        """Unit vector from the body straight at the wall."""
        return np.array([np.cos(np.radians(self.bearing_deg)), np.sin(np.radians(self.bearing_deg))])

    @property
    def foot(self) -> np.ndarray:
        """The wall's nearest point, relative to the middle of the body (same frame as the bearing)."""
        return self.distance_m * self.toward

    def __str__(self) -> str:
        return f"{self.distance_m:.2f} m at {self.bearing_deg:+.0f} deg (strength {self.strength:.2f})"


@dataclass
class Hearing:
    """What one ping told each ear, ready to be combined with other pings."""

    profiles: list[chirp.Profile]
    path_m: list[np.ndarray]        # per ear: the path axis (m, speaker -> that ear)
    reflectivity: list[np.ndarray]  # per ear: its profile along path_m, a perfect mirror = 1
    problems: list[str]             # what looked wrong with the recording, in words


@dataclass
class Look:
    """What Pupper worked out from one spot: one ping (four ears) or a few headings (two)."""

    echoes: list[Echo]          # strongest first, phantoms removed
    rings: list[Echo]           # peaks taken for the ring of a stronger echo. Not mapped, but
    #                             not walked into either: one of them may be a real wall
    dropped: list[Echo]         # peaks behind a wall: double bounces (kept for the tests)
    score: np.ndarray           # (ranges, bearings): how well a wall fits at each place
    floor: float                # the typical score: what "nothing there" looks like


def rotate(points: np.ndarray, angle: float) -> np.ndarray:
    """Turn (x, y) points counter-clockwise by angle (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return points @ np.array([[c, s], [-s, c]])


class Ears:
    def __init__(self, geometry: Geometry):
        self.geometry = geometry
        self.ranges = np.arange(MIN_RANGE_M, MAX_RANGE_M, RANGE_STEP_M)
        self.bearings = np.radians(np.arange(0.0, 360.0, BEARING_STEP_DEG))

    # ---------------------------------------------------------------- one ping
    def reflectivity(self, profile: chirp.Profile, ear: int) -> tuple[np.ndarray, np.ndarray]:
        """One ear's profile, rescaled so a perfect mirror reads 1 at any distance.

        Returns (path_m, reflectivity). path_m is the whole path from the
        speaker to this ear: the direct sound's (by ruler) plus the extra.

        Sound fades as 1/path. The profile is in units of the direct sound,
        which travelled direct_m; an echo off a perfect mirror that travelled
        `path` is direct_m / path as strong. So multiply by path / direct_m.
        """
        g = self.geometry
        path = g.direct_m[ear] + profile.extra_m
        value = profile.size * np.maximum(path, 0.0) / g.direct_m[ear]
        width = max(1, int(round(BLUR_M / chirp.SPEED_OF_SOUND * chirp.SAMPLE_RATE)))
        value = np.convolve(value, np.ones(width) / width, mode="same")
        if g.ceiling_m is not None:
            # The ceiling is a mirror too, and its echo reaches every ear at
            # nearly the same moment. Our ears sit too close together to be
            # sure that is "above" and not "a wall": cut it out. (The price:
            # a wall exactly as far away as the ceiling's echo goes unheard.)
            across = np.linalg.norm(g.mics[ear, :2] - g.speaker[:2])
            ceiling = np.hypot(across, 2 * g.ceiling_m - g.speaker[2] - g.mics[ear, 2])
            value = np.where(np.abs(path - ceiling) < CEILING_CUT_M, 0.0, value)
        return path, value

    def listen(self, recording: np.ndarray) -> Hearing:
        """A ping's recording, (samples, ears) -> each ear's profile."""
        g = self.geometry
        recording = np.asarray(recording, float)
        if recording.ndim != 2 or recording.shape[1] < len(g.mics):
            got = recording.shape[1] if recording.ndim == 2 else 1
            raise ValueError(f"mode '{g.mode}' needs {len(g.mics)} channels, the recording has {got}")
        profiles = [chirp.echo_profile(recording[:, k]) for k in range(len(g.mics))]
        curves = [self.reflectivity(p, k) for k, p in enumerate(profiles)]
        return Hearing(profiles, [c[0] for c in curves], [c[1] for c in curves], self.problems(recording, profiles))

    # ---------------------------------------------------------------- one look
    def wall_paths(self, heading: float, ear: int) -> np.ndarray:
        """For every wall the grid can hold: how far its echo travels to reach one ear.

        heading:  which way the nose pointed for this ping (radians).
        Returns:  (ranges, bearings). Entry [i, j] is for a wall ranges[i] from
                  the middle of the body, face-on in direction bearings[j].

        The image source is the speaker mirrored in the wall: start at the
        speaker and go twice the speaker's distance to the wall, straight at it.
        """
        g = self.geometry
        speaker = rotate(g.speaker[:2], heading)
        mic = rotate(g.mics[ear, :2], heading)
        d, b = np.meshgrid(self.ranges, self.bearings, indexing="ij")
        toward_x, toward_y = np.cos(b), np.sin(b)
        speaker_to_wall = np.maximum(d - (speaker[0] * toward_x + speaker[1] * toward_y), 0.0)
        image_x = speaker[0] + 2 * speaker_to_wall * toward_x
        image_y = speaker[1] + 2 * speaker_to_wall * toward_y
        return np.sqrt((image_x - mic[0]) ** 2 + (image_y - mic[1]) ** 2 + (g.speaker[2] - g.mics[ear, 2]) ** 2)

    def locate(self, pings: list[tuple[float, Hearing]]) -> Look:
        """Where the rings of all ears cross: the echoes of one look.

        pings:  [(heading in radians, Hearing)]: every ping made from this
                spot. One is enough for four ears. Two ears need a few
                headings, well apart: each heading has a phantom for every
                wall, in a different place each time, and only the real wall
                is there in all of them.
        """
        # "If a wall were here, did this ear hear it on this ping?" for every
        # wall the grid can hold, every ping, every ear.
        heard = [np.interp(self.wall_paths(heading, k), hearing.path_m[k], hearing.reflectivity[k], left=0.0, right=0.0)
                 for heading, hearing in pings for k in range(len(self.geometry.mics))]
        # All must agree: the geometric mean is small if any one heard nothing.
        score = np.exp(np.mean(np.log(np.stack(heard) + 1e-6), axis=0))
        floor = float(np.median(score))
        threshold = max(MIN_REFLECTIVITY, MIN_ABOVE_FLOOR * floor)

        peaks = []
        left = score.copy()
        n_ranges, n_bearings = score.shape
        range_reach = int(round(SAME_ECHO_M / RANGE_STEP_M))
        least_reach = int(round(SAME_ECHO_DEG / BEARING_STEP_DEG))
        while len(peaks) < MAX_PEAKS:
            i, j = np.unravel_index(int(np.argmax(left)), left.shape)
            top = left[i, j]
            if top < threshold:
                break
            # Walk along the ridge both ways from the peak, blanking it. With
            # ears close together a wall is a short arc, not a dot; without
            # this, one wall would be counted several times along its arc.
            reach = [0, 0]
            for side, step in enumerate((-1, 1)):
                row = i
                for n in range(1, n_bearings // 2):
                    column = (j + n * step) % n_bearings
                    rows = slice(max(0, row - 3), min(n_ranges, row + 4))
                    best = int(np.argmax(left[rows, column])) + rows.start
                    if n > least_reach and left[best, column] < RIDGE * top:
                        break
                    if left[best, column] >= RIDGE * top:
                        row, reach[side] = best, n
                    left[max(0, row - range_reach):row + range_reach + 1, column] = 0.0
            left[max(0, i - range_reach):i + range_reach + 1, j] = 0.0
            # Between grid points: a parabola through the neighbours, each way.
            fi = chirp.peak_position(score[:, j], i)
            fj = chirp.peak_position(np.take(score[i], np.arange(j - 1, j + 2), mode="wrap"), 1) - 1 + j
            peaks.append(Echo(
                bearing_deg=float((fj * BEARING_STEP_DEG + 180.0) % 360.0 - 180.0),
                distance_m=float(MIN_RANGE_M + fi * RANGE_STEP_M),
                strength=float(top),
                spread_deg=float(0.5 * (reach[0] + reach[1] + 1) * BEARING_STEP_DEG)))

        # Phantoms: rings first, then whatever is behind a wall. The second only
        # when every wall has one answer: two ears at a single heading still
        # have a mirror twin of every wall, and a twin must not be allowed to
        # hide a real wall.
        sure = not self.geometry.needs_turning or len({round(h, 2) for h, _ in pings}) > 1
        echoes, rings, dropped, real = [], [], [], []
        for peak in peaks:  # strongest first
            ring = any(abs(peak.distance_m - strong.distance_m) < RING_M and peak.strength < RING_SHARE * strong.strength
                       for strong in real)
            (rings if ring else real).append(peak)
        for echo in sorted(real, key=lambda e: e.distance_m):
            hidden = sure and any(solid.strength >= SOLID_STRENGTH
                                  and float(np.dot(echo.foot, solid.toward)) > solid.distance_m + BEHIND_M
                                  for solid in echoes)
            (dropped if hidden else echoes).append(echo)
        echoes.sort(key=lambda e: -e.strength)
        return Look(echoes, rings, dropped, score, floor)

    def hear(self, recording: np.ndarray, heading: float = 0.0) -> Look:
        """One ping, one look: the short way for four ears (and for mic_check.py)."""
        return self.locate([(heading, self.listen(recording))])

    # ---------------------------------------------------------------- is the recording any good?
    def problems(self, recording: np.ndarray, profiles: list[chirp.Profile]) -> list[str]:
        """What looks wrong with this recording, said so a person can fix it."""
        names = self.geometry.names
        found = []
        for name, p in zip(names, profiles):
            if p.clipped:
                found.append(f"{name}: too loud, the recording clips. Turn the speaker or the mic gain down.")
            if p.noise > 0.05:   # the profile is in units of the direct sound: 0.05 = noise only 20 times quieter
                found.append(f"{name}: can barely hear the chirp over the noise. Speaker volume up? Mic on and unmuted?")
            if p.steadiness < 0.7:
                found.append(f"{name}: the chirps were not equally loud ({p.steadiness:.0%}). "
                             "Turn off auto gain and noise cancelling on the receiver.")
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                x, y = recording[:, a], recording[:, b]
                same = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)
                if same > 0.995:
                    found.append(f"{names[a]} and {names[b]} recorded the same thing: the receiver is in Mono. "
                                 "Set it to Stereo (or Quadraphonic), so each mic gets its own channel.")
        return found
