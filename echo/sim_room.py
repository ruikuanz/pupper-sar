"""A pretend room: what the ears would record if Pupper pinged in it.

This is the laptop's stand-in for the speaker, the air, the walls and the
microphones. It hands back a recording with the same shape as the real sound
card's, so everything after it (chirp.py, ears.py, room_map.py, navigator.py)
runs unchanged, and the tests can ask "is the wall where you say it is?".

How sound gets from the speaker to an ear, in this model:

  - straight there (the direct sound);
  - off one wall. A flat wall is a mirror for sound: the echo arrives exactly
    as if a second speaker stood behind the wall, at the mirror image of the
    real one (the *image source*). The echo's path is the straight line from
    the image to the ear, and it only exists if that line crosses the wall
    where the wall really is, and nothing else is in the way;
  - off two walls (the image of an image). This is how a corner answers:
    sound goes in, bounces off both walls and comes straight back;
  - each of those also off the floor or off the ceiling;
  - off a post (a chair leg, a person's shin): weaker, in every direction;
  - and then the room's ringing: thousands of later, weaker, random echoes
    (`clutter`), which is what limits how far the ears can hear a wall.

Sound gets quieter as 1/distance, and loses a little at every bounce.

It is honest about what the real hardware does not promise: every ear gets its
own unknown delay on every ping, the speaker colors the sound, and there is
hiss.

Numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import chirp

MAX_PATH_M = 14.0      # echoes longer than this are left to the clutter
CLUTTER_FROM_S = 0.006  # the ringing starts this long after the sound leaves
CLUTTER_UNTIL_S = 0.2


@dataclass
class Wall:
    a: tuple[float, float]   # one end (m, world)
    b: tuple[float, float]   # the other end
    reflect: float = 0.85    # share of the sound pressure that bounces (painted wall 0.95, cardboard 0.8, curtain 0.3)


@dataclass
class Post:
    at: tuple[float, float]
    radius: float = 0.05     # m


def mirror(p: np.ndarray, wall: Wall) -> np.ndarray:
    """The mirror image of point p in the wall's line."""
    a, b = np.asarray(wall.a, float), np.asarray(wall.b, float)
    along = (b - a) / np.linalg.norm(b - a)
    foot = a + np.dot(p - a, along) * along
    return 2 * foot - p


def crossing(p: np.ndarray, q: np.ndarray, wall: Wall) -> tuple[float, float] | None:
    """Where the segment p->q crosses the wall: (fraction along p->q, fraction along the wall), or None."""
    a, b = np.asarray(wall.a, float), np.asarray(wall.b, float)
    d, e = q - p, b - a
    cross = d[0] * e[1] - d[1] * e[0]
    if abs(cross) < 1e-12:
        return None
    t = ((a[0] - p[0]) * e[1] - (a[1] - p[1]) * e[0]) / cross
    u = ((a[0] - p[0]) * d[1] - (a[1] - p[1]) * d[0]) / cross
    return (t, u) if 0.0 <= u <= 1.0 and 1e-9 < t < 1.0 - 1e-9 else None


@dataclass
class SimRoom:
    walls: list[Wall]
    posts: list[Post] = field(default_factory=list)
    ceiling_m: float | None = 2.7    # None: outdoors
    floor_reflect: float = 0.5       # carpet 0.3, wood or tile 0.9
    ceiling_reflect: float = 0.7
    clutter: float = 1.0             # 0: an empty hall. 1: a furnished room. 2: a cluttered one
    hiss: float = 0.002              # microphone noise, of full scale
    volume: float = 0.02             # full-scale chirp -> this recording level at 1 m
    seed: int = 0

    def __post_init__(self) -> None:
        self.random = np.random.default_rng(self.seed)

    # ---------------------------------------------------------------- handy rooms
    @classmethod
    def box(cls, width: float, depth: float, **more) -> "SimRoom":
        """A rectangular room: x from 0 to width, y from 0 to depth."""
        c = [(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)]
        return cls([Wall(c[i], c[(i + 1) % 4]) for i in range(4)], **more)

    @classmethod
    def polygon(cls, corners: list[tuple[float, float]], **more) -> "SimRoom":
        n = len(corners)
        return cls([Wall(corners[i], corners[(i + 1) % n]) for i in range(n)], **more)

    # ---------------------------------------------------------------- geometry
    def clear(self, p: np.ndarray, q: np.ndarray, skip: tuple[int, ...] = ()) -> bool:
        """Can sound go straight from p to q (ignoring the walls in skip)?"""
        return not any(crossing(p, q, w) for i, w in enumerate(self.walls) if i not in skip)

    def floor_paths(self, speaker: np.ndarray, ear: np.ndarray) -> list[tuple[float, float]]:
        """Every path from the speaker to one ear, seen from above: (length m, gain from bounces).

        speaker, ear: (x, y) in the room. Heights are added in paths().
        """
        found = []
        if self.clear(speaker, ear):
            found.append((float(np.linalg.norm(ear - speaker)), 1.0))
        for i, w1 in enumerate(self.walls):
            image1 = mirror(speaker, w1)
            hit = crossing(image1, ear, w1)
            if hit:
                point = image1 + hit[0] * (ear - image1)
                if self.clear(speaker, point, (i,)) and self.clear(point, ear, (i,)):
                    found.append((float(np.linalg.norm(ear - image1)), w1.reflect))
            for j, w2 in enumerate(self.walls):
                if j == i:
                    continue
                image2 = mirror(image1, w2)
                length = float(np.linalg.norm(ear - image2))
                if length > MAX_PATH_M:
                    continue
                hit2 = crossing(image2, ear, w2)
                if not hit2:
                    continue
                point2 = image2 + hit2[0] * (ear - image2)
                hit1 = crossing(image1, point2, w1)
                if not hit1:
                    continue
                point1 = image1 + hit1[0] * (point2 - image1)
                if (self.clear(speaker, point1, (i,)) and self.clear(point1, point2, (i, j))
                        and self.clear(point2, ear, (j,))):
                    found.append((length, w1.reflect * w2.reflect))
        for post in self.posts:
            at = np.asarray(post.at, float)
            out, back = np.linalg.norm(at - speaker), np.linalg.norm(ear - at)
            if out > post.radius and back > post.radius and self.clear(speaker, at) and self.clear(at, ear):
                # A round post sends sound back in every direction: at the ear it is
                # sqrt(radius / 2) / (out * sqrt(back)) strong. (paths() divides by
                # the length, as for a mirror, so multiply by it here.)
                found.append((float(out + back), float(np.sqrt(post.radius / 2) / (out * np.sqrt(back)) * (out + back))))
        return found

    def paths(self, speaker: np.ndarray, ear: np.ndarray) -> list[tuple[float, float]]:
        """Every path from the speaker to one ear: (length m, strength at the ear).

        speaker, ear: (x, y, z) in the room, z up from the floor.
        """
        found = []
        for length, gain in self.floor_paths(speaker[:2], ear[:2]):
            heights = [(ear[2] - speaker[2], 1.0), (ear[2] + speaker[2], self.floor_reflect)]
            if self.ceiling_m is not None:
                heights.append((2 * self.ceiling_m - ear[2] - speaker[2], self.ceiling_reflect))
            for rise, bounce in heights:
                total = float(np.hypot(length, rise))
                found.append((total, gain * bounce / total))
        return found

    def ringing(self, n: int) -> np.ndarray:
        """The room's late, random echoes, as n samples of impulse response."""
        response = np.zeros(n)
        if self.clutter <= 0:
            return response
        # In a room, echoes arrive more and more densely (as time squared), each
        # one weaker (spreading, and a loss at every bounce).
        count = int(4000 * self.clutter)
        u = self.random.random(count)
        times = (CLUTTER_FROM_S ** 3 + u * (CLUTTER_UNTIL_S ** 3 - CLUTTER_FROM_S ** 3)) ** (1 / 3)
        travelled = chirp.SPEED_OF_SOUND * times
        strength = 0.5 * 0.7 ** (travelled / 2.5) / travelled * self.random.choice([-1.0, 1.0], count)
        np.add.at(response, np.minimum((times * chirp.SAMPLE_RATE).astype(int), n - 1), strength)
        return response

    # ---------------------------------------------------------------- the recording
    def record(self, pose: tuple[float, float, float], speaker: np.ndarray, mics: np.ndarray) -> np.ndarray:
        """Ping once from `pose` and return what the ears recorded.

        pose:     (x, y, heading in radians) of the dog's body in the room.
        speaker:  (3,) and mics: (n, 3): where they are on the body (body frame:
                  x forward, y left, z up from the floor).
        Returns:  (samples, n) float array, like the sound card's.
        """
        x, y, heading = pose
        cos, sin = np.cos(heading), np.sin(heading)

        def to_room(p: np.ndarray) -> np.ndarray:
            return np.array([x + cos * p[0] - sin * p[1], y + sin * p[0] + cos * p[1], p[2]])

        ping = chirp.make_ping()
        n = len(ping) + int(0.08 * chirp.SAMPLE_RATE)
        size = 1 << int(np.ceil(np.log2(n)))
        freqs = np.fft.rfftfreq(size, 1.0 / chirp.SAMPLE_RATE)
        # A small speaker: a resonance near 6 kHz colors everything it plays.
        f = np.maximum(freqs, 1.0)
        speaker_color = 1.0 / (1.0 + 1.0j * (f / 6000.0 - 6000.0 / f))
        played = np.fft.rfft(ping, size) * speaker_color * self.volume

        source = to_room(np.asarray(speaker, float))
        chain_delay = self.random.uniform(0.02, 0.06)   # playing and recording are not in step
        recording = np.zeros((n, len(mics)))
        for k, mic in enumerate(np.asarray(mics, float)):
            delay = chain_delay + self.random.uniform(-0.003, 0.003)  # ... and neither are the ears
            air = np.zeros(len(freqs), complex)
            for length, strength in self.paths(source, to_room(mic)):
                air += strength * np.exp(-2j * np.pi * freqs * (length / chirp.SPEED_OF_SOUND + delay))
            ring = np.fft.rfft(self.ringing(size), size) * np.exp(-2j * np.pi * freqs * delay)
            heard = np.fft.irfft(played * (air + ring), size)[:n]
            recording[:, k] = heard + self.hiss * self.random.standard_normal(n)
        return np.clip(recording, -1.0, 1.0)

    # ---------------------------------------------------------------- for the tests and the map page
    def distance_to_wall(self, point: tuple[float, float]) -> float:
        """How far the nearest wall or post is from a point."""
        p = np.asarray(point, float)
        best = np.inf
        for w in self.walls:
            a, b = np.asarray(w.a, float), np.asarray(w.b, float)
            t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
            best = min(best, float(np.linalg.norm(p - (a + t * (b - a)))))
        for post in self.posts:
            best = min(best, float(np.linalg.norm(p - np.asarray(post.at))) - post.radius)
        return best
