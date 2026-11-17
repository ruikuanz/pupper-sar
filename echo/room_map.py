"""The map: where the walls are in the room, and where Pupper is among them.

Everything is in the *room frame*: metres, with the origin where Pupper stood
when it started, x the way its nose pointed then, y to its left.

What goes in
------------
Each look (ears.py) gives echoes: "a wall, face-on, D metres away in direction
B". Put at Pupper's position, each one is a Sighting: the wall's nearest point
(its foot), and which way the wall faces.

What the map makes of them
--------------------------
A flat wall answers from wherever Pupper stands, each time from the point
nearest to Pupper. A corner (or a post) answers from the same point wherever
Pupper stands. A Landmark is either, and Landmark.refit tells which from how
the distance to it changes as Pupper moves about.

A landmark is *confirmed* once it has been seen from two different spots.
Clutter, and the odd phantom that got past ears.py, rarely turn up in the
same place twice; landmarks that are never seen again are forgotten.

Where am I? (fix)
-----------------
Walking is not exact: Pupper thinks it went 0.5 m and went 0.42. But a
confirmed wall is where it is. If the map says that wall should be 1.30 m away
and the echo says 1.38 m, Pupper is 8 cm further from it than it thought. One
wall corrects the position along one direction; two walls at an angle correct
all of it. This is why a blind robot can keep walking: every look pulls its
idea of where it is back to the walls.

The heading is not corrected here: it comes from the gyro, which is much
better at angles than two or four ears a hand apart.

How far can I go that way? (clearance)
--------------------------------------
Every echo is a wall face-on. Pupper assumes it runs on forever to both sides,
which is the careful assumption: walking any direction, it reaches that wall's
plane after D / cos(angle between the direction and the wall). Weak echoes are
posts: something to steer around, not a wall across the room.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ears import BEHIND_M, SOLID_STRENGTH, Echo

# --- is this the same wall? -----------------------------------------------------
# Distances from echoes are good to a couple of centimetres. Bearings are not:
# with ears a hand apart, 10 degrees off is ordinary, 30 happens. So a sighting
# belongs to a landmark if its *distance* is what the landmark predicts, and
# its bearing is merely in the right part of the room.
SAME_RANGE_M = 0.15      # a sighting of a known landmark is this close to the distance it predicts
SAME_FACING_DEG = 35.0   # ... and faces within this of where it predicts
NEW_SPOT_M = 0.15        # a look counts as "another spot" if Pupper moved at least this far
FORGET_AFTER = 4         # looks an unconfirmed landmark survives without being seen again
PATCH_M = 0.25           # a wall is drawn (and trusted) this far beyond where it has been seen
SETTLED_SPOTS = 3        # a wall seen from this many spots is trusted to hide what is behind it
YOUNG = 4                # a landmark with fewer sightings than this may still turn out to be a corner

# --- wall or corner? (Landmark.refit) -----------------------------------------------
FIT_SPREAD_M = 0.3       # spots must be this far apart before distances alone can say
FIT_TURN_DEG = 40        # the wall's facing is searched this far either side of how it sounded
FIT_BEARING_DEG = 15.0   # how far off a wall's facing may sound ...
FIT_RANGE_M = 0.03       # ... weighed against distances good to this
FIT_BETTER_M = 0.02      # a corner must fit the distances better than a wall by this much
CORNER_FACING_DEG = 35.0  # before that: heard from directions this far apart, it is a corner

IN_FRONT_M = 0.15        # tidy: every spot Pupper stood on is at least this far in front of a real wall
CORNER_NEAR_M = 0.3      # tidy: a "wall" this close to (or beyond) two other walls' lines ...
CORNER_SLANT_DEG = 20.0  # ... and slanted at least this much to both, is their corner

# --- where am I? ----------------------------------------------------------------
# An echo is matched to a landmark in two passes: first only echoes that face
# the way the landmark does (a corner at the same distance as a wall must not
# pass for it), then, from the corrected position, looser on facing and
# stricter on distance.
MATCH_PASSES = ((0.30, 15.0), (0.15, 30.0))   # (distance gate m, facing gate deg)
ODD_ONE_OUT_M = 0.12     # after solving, an echo that still disagrees by this much was a wrong match
MAX_FIX_M = 0.3          # never jump further than this on one look
CORNER_WEIGHT = 0.3      # corners count less than walls: their bearing is shakier
# How wrong each thing may be (one standard deviation). The fix weighs them against each other.
ECHO_SIGMA_M = 0.05      # an echo's distance against the map (the map is not perfect either)
ALONG_SHARE = 0.25       # a step's length: a quarter of it. Dogs walk short, or long.
ALONG_LEAST_M = 0.05
ACROSS_SHARE = 0.08      # a step's direction: the gyro holds it, so sideways it lands close
ACROSS_LEAST_M = 0.03

# --- how far can I go? ------------------------------------------------------------
WEAK_HALF_M = 0.5        # a weak echo blocks this far to each side of its point
FAR_M = 9.0              # "no wall that way" reads as this


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Degrees between two unit vectors."""
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def unit(deg: float) -> np.ndarray:
    return np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])


@dataclass
class Sighting:
    foot: np.ndarray     # the wall's nearest point to Pupper, in the room (as it sounded)
    toward: np.ndarray   # unit vector from Pupper straight at the wall (as it sounded)
    distance: float      # from Pupper to the wall: the part of an echo to trust
    strength: float
    spot: np.ndarray     # where Pupper stood
    look: int            # which look this was


@dataclass
class Landmark:
    """A wall (a line) or a corner/post (a point), and every sighting of it."""

    sightings: list[Sighting] = field(default_factory=list)
    last_look: int = 0
    # Worked out from the sightings by refit():
    is_corner: bool = False
    toward: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))   # a wall faces this way
    offset: float = 0.0          # a wall's line is every point p with toward . p = offset
    point: np.ndarray = field(default_factory=lambda: np.zeros(2))             # a corner is here
    heard_at: np.ndarray = field(default_factory=lambda: np.zeros(2))          # the average of where it sounded to be

    def refit(self) -> None:
        """Work out what this landmark is and where, from all its sightings.

        At first all we have is how it sounded: average the bearings.

        Once Pupper has heard it from spots well apart, the distances alone
        say more than any bearing, and say what it is:

            a wall     its distance changes by exactly how far Pupper moved
                       toward it:      distance = offset - toward . spot
            a corner   its distance is the distance to one point:
                                       distance = |point - spot|

        Fit both; keep the one the distances agree with. This is the same
        triangulation as between the ears, with the ears replaced by the
        places Pupper stood: metres apart instead of centimetres, so much
        sharper.
        """
        weights = np.array([s.strength for s in self.sightings])
        feet = np.array([s.foot for s in self.sightings])
        spots = np.array([s.spot for s in self.sightings])
        distances = np.array([s.distance for s in self.sightings])
        sounded = (weights[:, None] * np.array([s.toward for s in self.sightings])).sum(axis=0)
        sounded /= max(np.linalg.norm(sounded), 1e-9)
        # How it sounded.
        self.toward = sounded
        self.offset = float(np.dot(weights, feet @ sounded) / weights.sum())
        self.point = self.heard_at = (weights[:, None] * feet).sum(axis=0) / weights.sum()
        self.is_corner = max(angle_between(sounded, s.toward) for s in self.sightings) > CORNER_FACING_DEG
        spread = max(np.linalg.norm(a - b) for a in spots for b in spots)
        if spread < FIT_SPREAD_M:
            return
        # As a wall: try every facing near how it sounded; the offset follows.
        # How it sounded still counts for something (FIT_BEARING_DEG): two
        # spots in line with the wall say nothing about its facing, and a
        # step that fell short would otherwise be read as a slanted wall.
        middle = np.degrees(np.arctan2(sounded[1], sounded[0]))
        best_wall = (np.inf, 0.0, sounded, self.offset)
        for turn in np.arange(-FIT_TURN_DEG, FIT_TURN_DEG + 1, 1.0):
            n = unit(middle + turn)
            offsets = distances + spots @ n
            miss = float(offsets.std())
            cost = miss ** 2 + (turn / FIT_BEARING_DEG * FIT_RANGE_M) ** 2
            if cost < best_wall[0]:
                best_wall = (cost, miss, n, float(offsets.mean()))
        if len(self.sightings) < 3:
            _, _, self.toward, self.offset = best_wall   # two spots can straighten a wall, not tell it from a corner
            return
        # As a corner: start where it sounded, and step toward the point that fits (Gauss-Newton).
        point = self.point.copy()
        for _ in range(12):
            to = point - spots
            reach = np.maximum(np.linalg.norm(to, axis=1), 1e-6)
            slope = to / reach[:, None]
            step = np.linalg.lstsq(slope, distances - reach, rcond=None)[0]
            point = point + np.clip(step, -0.3, 0.3)
        corner_miss = float(np.sqrt(np.mean((np.linalg.norm(point - spots, axis=1) - distances) ** 2)))
        self.is_corner = corner_miss + FIT_BETTER_M < best_wall[1]
        if self.is_corner:
            self.point = point
        else:
            _, _, self.toward, self.offset = best_wall

    # ------------------------------------------------------------ what it is
    @property
    def strength(self) -> float:
        return float(np.mean([s.strength for s in self.sightings]))

    @property
    def spots(self) -> int:
        """From how many different spots it has been seen."""
        seen: list[np.ndarray] = []
        for s in self.sightings:
            if all(np.linalg.norm(s.spot - other) >= NEW_SPOT_M for other in seen):
                seen.append(s.spot)
        return len(seen)

    @property
    def confirmed(self) -> bool:
        return self.spots >= 2

    @property
    def ends(self) -> tuple[np.ndarray, np.ndarray]:
        """A wall, as far as it has been seen to run: its two ends.

        A wall answers from the point nearest to Pupper, so it has been seen
        wherever Pupper's spots fall on its line.
        """
        along = np.array([-self.toward[1], self.toward[0]])
        t = [float(np.dot(along, s.spot)) for s in self.sightings]
        base = self.offset * self.toward
        return base + (min(t) - PATCH_M) * along, base + (max(t) + PATCH_M) * along

    # ------------------------------------------------------------ what should Pupper hear of it?
    def expect(self, spot: np.ndarray) -> tuple[np.ndarray, float]:
        """From `spot`: (unit vector straight at it, distance)."""
        if self.is_corner:
            to = self.point - spot
            distance = float(np.linalg.norm(to))
            return to / max(distance, 1e-9), distance
        return self.toward, float(self.offset - np.dot(self.toward, spot))

    def fits(self, s: Sighting) -> bool:
        """Is this sighting one of mine? Its distance is what I predict, its bearing roughly so."""
        toward, distance = self.expect(s.spot)
        if abs(distance - s.distance) < SAME_RANGE_M and angle_between(toward, s.toward) < SAME_FACING_DEG:
            return True
        if self.is_corner or len(self.sightings) >= YOUNG:
            return False
        # While young, a landmark filed as a wall may be a corner: also take a
        # sighting that fits the *point* where it was heard. Three of those
        # and refit can tell which it is.
        to = self.heard_at - s.spot
        reach = float(np.linalg.norm(to))
        return abs(reach - s.distance) < SAME_RANGE_M and angle_between(to / max(reach, 1e-9), s.toward) < SAME_FACING_DEG


class RoomMap:
    def __init__(self) -> None:
        self.landmarks: list[Landmark] = []
        self.looks = 0
        self.spots: list[np.ndarray] = []   # where Pupper has looked from

    # ---------------------------------------------------------------- where am I?
    def fix(self, spot: np.ndarray, echoes: list[Echo], step: np.ndarray | None = None) -> tuple[np.ndarray, int]:
        """Correct Pupper's position with the walls it already knows.

        spot:    (x, y) where Pupper thinks it is (from counting its steps).
        echoes:  this look's, bearings in the room frame.
        step:    (dx, dy) it thinks it walked since the last look.
        Returns: (the corrected position, how many echoes were matched to the map).

        Each matched echo says "you are `expected - heard` metres nearer to
        this wall than you thought": one equation, toward . move = expected -
        heard. Solve them all together (least squares), each weighed by how
        far it can be trusted.

        The step is in there too, and it is not equally unsure in every
        direction. How *far* Pupper walked is a guess (legs slip). Which *way*
        is known from the gyro. So the position may slide a good way along the
        step and only a little across it. This matters: early on the map's
        walls come from bearings alone and can be tilted by ten degrees, and a
        fix that trusted them fully would bend Pupper's path to match the
        tilt. Holding the direction lets Landmark.refit straighten the wall
        from the distances instead.
        """
        start = np.asarray(spot, float)
        known = [m for m in self.landmarks if m.confirmed]
        if not known:
            # Second look ever: nothing is confirmed yet. Trust the solid walls of the first.
            known = [m for m in self.landmarks if m.strength >= SOLID_STRENGTH and not m.is_corner]
        # What the step alone says: stay put, give or take more along the step than across it.
        length = 0.0 if step is None else float(np.linalg.norm(step))
        along = np.array([1.0, 0.0]) if length < 1e-6 else np.asarray(step, float) / length
        across = np.array([-along[1], along[0]])
        prior = (np.outer(along, along) / max(ALONG_LEAST_M, ALONG_SHARE * length) ** 2
                 + np.outer(across, across) / max(ACROSS_LEAST_M, ACROSS_SHARE * length) ** 2)
        matched = 0
        move = np.zeros(2)
        for gate_m, gate_deg in MATCH_PASSES:
            # Pair each echo with the landmark it fits best, one echo per landmark.
            options = []
            for i, echo in enumerate(echoes):
                for j, mark in enumerate(known):
                    toward, expected = mark.expect(start + move)
                    apart, turned = abs(echo.distance_m - expected), angle_between(toward, echo.toward)
                    if apart < gate_m and turned < gate_deg:
                        weight = echo.strength * (CORNER_WEIGHT if mark.is_corner else 1.0)
                        # The equation is written from `start`: toward . move = (distance expected at start) - heard.
                        want = mark.expect(start)[1] - echo.distance_m
                        options.append((apart / gate_m + turned / gate_deg, i, j, toward, want, weight))
            pairs, used_echoes, used_marks = [], set(), set()
            for _, i, j, toward, want, weight in sorted(options, key=lambda o: o[0]):
                if i not in used_echoes and j not in used_marks:
                    used_echoes.add(i), used_marks.add(j)
                    pairs.append((toward, want, weight))
            # Solve; if one echo still disagrees with the rest, it was a wrong match: drop it, solve again.
            while pairs:
                a = np.array([p[0] for p in pairs])
                b = np.array([p[1] for p in pairs])
                w = np.array([p[2] for p in pairs])
                w = w / ECHO_SIGMA_M ** 2
                move = np.linalg.solve((a * w[:, None]).T @ a + prior, (a * w[:, None]).T @ b)
                left_over = np.abs(a @ move - b)
                worst = int(np.argmax(left_over))
                if left_over[worst] < ODD_ONE_OUT_M or len(pairs) < 3:
                    break
                pairs.pop(worst)
            if pairs:
                matched = len(pairs)
            elif not matched:
                move = np.zeros(2)
        if np.linalg.norm(move) > MAX_FIX_M:
            move *= MAX_FIX_M / np.linalg.norm(move)
        return start + move, matched

    # ---------------------------------------------------------------- add a look
    def add(self, spot: np.ndarray, echoes: list[Echo]) -> None:
        """Put one look's echoes on the map. spot: (x, y); bearings in the room frame."""
        spot = np.asarray(spot, float)
        self.looks += 1
        self.spots.append(spot)
        settled = [m for m in self.landmarks if m.spots >= SETTLED_SPOTS and not m.is_corner and m.strength >= SOLID_STRENGTH]
        for echo in echoes:
            sighting = Sighting(spot + echo.foot, echo.toward, echo.distance_m, echo.strength, spot, self.looks)
            # Behind a wall the map is sure of (and Pupper in front of it): a
            # double bounce that ears.py could not catch this time.
            if any(np.dot(m.toward, sighting.foot) > m.offset + 2 * BEHIND_M and np.dot(m.toward, spot) < m.offset
                   and not m.fits(sighting) for m in settled):
                continue
            # The landmark it fits best (by distance), if any.
            home = min((m for m in self.landmarks if m.fits(sighting)),
                       key=lambda m: abs(m.expect(spot)[1] - sighting.distance), default=None)
            if home is None:
                home = Landmark()
                self.landmarks.append(home)
            home.sightings.append(sighting)
            home.last_look = self.looks
            home.refit()
        self.landmarks = [m for m in self.landmarks if m.confirmed or self.looks - m.last_look < FORGET_AFTER]
        self.tidy()

    def tidy(self) -> None:
        """Spot the corners that still pass for walls.

        A corner answers like a short wall set diagonally across it, and until
        Pupper has moved far enough for the distances to tell (refit), that is
        how it sits on the map. But the map knows better: a "wall" lying where
        two confirmed walls meet, at a slant to both, is their corner. Put it
        where the two lines cross, which is known better than any bearing.
        """
        # First: Pupper cannot have stood behind a wall. A "wall" with one of
        # Pupper's own spots well beyond it is a phantom.
        self.landmarks = [m for m in self.landmarks if m.is_corner
                          or all(np.dot(m.toward, spot) < m.offset - IN_FRONT_M for spot in self.spots)]
        walls = [m for m in self.landmarks if m.confirmed and not m.is_corner]
        for mark in walls:
            a, b = mark.ends
            middle = 0.5 * (a + b)
            meeting = [w for w in walls if w is not mark
                       and np.dot(w.toward, middle) - w.offset > -CORNER_NEAR_M
                       and CORNER_SLANT_DEG < angle_between(w.toward, mark.toward) < 90.0 - CORNER_SLANT_DEG]
            if len(meeting) >= 2:
                first, second = meeting[0], meeting[1]
                lines = np.array([first.toward, second.toward])
                if abs(np.linalg.det(lines)) > 0.2:   # the two walls are not parallel
                    mark.is_corner = True
                    mark.point = np.linalg.solve(lines, np.array([first.offset, second.offset]))

    # ---------------------------------------------------------------- the room's outline
    def outline(self, around: np.ndarray, reach: float = 6.0) -> list[tuple[float, float]]:
        """The room, if every confirmed wall ran on to meet its neighbours.

        Start with a big square around `around` and cut it with the line of
        each solid wall heard from three spots, keeping the side Pupper is
        on. In a room without nooks this is the floor plan, usually after
        three or four looks (each look hears every wall, wherever Pupper
        stands).
        """
        around = np.asarray(around, float)
        shape = [around + reach * np.array(c, float) for c in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        for mark in self.landmarks:
            if mark.spots < SETTLED_SPOTS or mark.is_corner or mark.strength < SOLID_STRENGTH:
                continue  # only walls heard from three spots: one stray must not cut the room in half
            n, offset = mark.toward, mark.offset
            if np.dot(n, around) > offset:
                continue  # Pupper is behind this "wall": not one to cut the room with
            cut = []
            for i, p in enumerate(shape):
                q = shape[(i + 1) % len(shape)]
                p_in, q_in = np.dot(n, p) <= offset, np.dot(n, q) <= offset
                if p_in:
                    cut.append(p)
                if p_in != q_in:
                    t = (offset - np.dot(n, p)) / np.dot(n, q - p)
                    cut.append(p + t * (q - p))
            shape = cut
            if len(shape) < 3:
                return []
        return [(float(p[0]), float(p[1])) for p in shape]

    # ---------------------------------------------------------------- how far can I go?
    def clearance(self, spot: np.ndarray, direction_deg: float, echoes: list[Echo]) -> float:
        """How far the middle of the body can travel from `spot` in a direction before touching something.

        echoes: the look just made from this spot (bearings in the room frame).
          - a solid echo is a wall, taken to run on forever both ways;
          - a weak echo is a post or a poor reflector: a wall WEAK_HALF_M to each side;
          - confirmed landmarks count too, where they have been seen to run:
            a wall the last look happened to miss is still there.
        """
        spot = np.asarray(spot, float)
        heading = np.array([np.cos(np.radians(direction_deg)), np.sin(np.radians(direction_deg))])
        best = FAR_M

        def wall(toward: np.ndarray, distance: float, middle: np.ndarray | None = None, half: float = 0.0) -> None:
            """A wall facing `toward`, `distance` away; if `middle` is given it only runs `half` either side of it."""
            nonlocal best
            closing = float(np.dot(heading, toward))   # how much of each step goes toward the wall
            if closing <= 0.05 or distance < -0.1:
                return
            reach = max(distance, 0.0) / closing
            if middle is not None:
                hit = spot + reach * heading
                if abs(float(np.dot(hit - middle, [-toward[1], toward[0]]))) > half:
                    return
            best = min(best, reach)

        for echo in echoes:
            if echo.strength >= SOLID_STRENGTH:
                wall(echo.toward, echo.distance_m)
            else:
                wall(echo.toward, echo.distance_m, spot + echo.foot, WEAK_HALF_M)
        for mark in self.landmarks:
            if not mark.confirmed:
                continue
            toward, distance = mark.expect(spot)
            if mark.is_corner or mark.strength < SOLID_STRENGTH:
                wall(toward, distance, mark.point, WEAK_HALF_M)
            else:
                a, b = mark.ends
                wall(toward, distance, 0.5 * (a + b), 0.5 * float(np.linalg.norm(b - a)))
        return best

    # ---------------------------------------------------------------- for the map page
    def as_dict(self) -> dict:
        walls, corners = [], []
        for mark in self.landmarks:
            if mark.is_corner:
                p = mark.point
                corners.append({"at": [float(p[0]), float(p[1])], "confirmed": mark.confirmed, "strength": mark.strength})
            else:
                a, b = mark.ends
                walls.append({"a": [float(a[0]), float(a[1])], "b": [float(b[0]), float(b[1])],
                              "confirmed": mark.confirmed, "strength": mark.strength,
                              "feet": [[float(s.foot[0]), float(s.foot[1])] for s in mark.sightings]})
        return {"walls": walls, "corners": corners, "spots": [[float(s[0]), float(s[1])] for s in self.spots]}
