"""Desk-side tests for the echolocation component: no robot, no ROS, no microphones.

    python3 tests/test_echo.py             # everything (about half a minute)
    python3 tests/test_echo.py ears        # one part: chirp, ears, map, navigator, tools

Like the labs' test.py: every check prints what it wanted and what it got.
The sound comes from echo/sim_room.py: a pretend room that hands back what
the microphones would record, with the delays, noise and ringing that real
ones have. The pretend dog walks short and its ears are not quite where
ears.yaml says, as on a real one; the tests then ask whether the walls it
found are where the room's walls are.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "echo"))

import chirp  # noqa: E402
import ears as E  # noqa: E402
import mic_check  # noqa: E402
import navigator as N  # noqa: E402
import room_map as M  # noqa: E402
import sound as S  # noqa: E402
from dog import SimDog  # noqa: E402
from sim_room import Post, SimRoom  # noqa: E402

FAILED = []
COUNT = [0]


def check(name: str, ok: bool, got=None, want=None) -> None:
    COUNT[0] += 1
    if ok:
        print(f"  ok    {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}\n          got:    {got}\n          wanted: {want}")


def geometry(mode: str) -> E.Geometry:
    return E.load_geometry(mode, calibration=None)   # the tests never read a robot's calibration.json


def off_by(echo: E.Echo, bearing_deg: float) -> float:
    return abs(N.wrap(echo.bearing_deg - bearing_deg))


def find(echoes: list[E.Echo], bearing_deg: float, distance: float, within_deg: float = 15.0) -> E.Echo | None:
    """The echo nearest in distance among those in roughly that direction."""
    near = [e for e in echoes if off_by(e, bearing_deg) < within_deg]
    return min(near, key=lambda e: abs(e.distance_m - distance), default=None)


# ------------------------------------------------------------------ chirp
def test_chirp() -> None:
    print("chirp: finding the direct sound and an echo in one ear's recording")
    ping = chirp.make_ping()
    rate, c = chirp.SAMPLE_RATE, chirp.SPEED_OF_SOUND
    rng = np.random.default_rng(0)

    def heard(delay_s: float, echo_m: float, echo_size: float, noise: float = 0.002) -> np.ndarray:
        n = len(ping) + int(0.1 * rate)
        spectrum = np.fft.rfft(ping, n)
        freqs = np.fft.rfftfreq(n, 1 / rate)
        air = np.exp(-2j * np.pi * freqs * delay_s) + echo_size * np.exp(-2j * np.pi * freqs * (delay_s + echo_m / c))
        return 0.3 * np.fft.irfft(spectrum * air, n) + noise * rng.standard_normal(n)

    for delay in (0.0213, 0.0577):   # the unknown delay of sound cards and radio: must not matter
        p = chirp.echo_profile(heard(delay, 2.0, 0.1))
        after = p.extra_m > 0.3
        at = p.extra_m[after][np.argmax(p.size[after])]
        check(f"echo 2.000 m after the direct sound, with {delay * 1000:.1f} ms of unknown delay",
              abs(at - 2.0) < 0.01, f"{at:.3f} m", "2.000 m +- 0.01")
        check("  ... a tenth as strong as the direct sound", abs(p.size[after].max() - 0.1) < 0.02, round(float(p.size[after].max()), 3), 0.1)
    width = np.sum(p.size[np.abs(p.extra_m) < 0.5] > 0.5) * c / rate
    check("the 8 ms chirp is squeezed to a spike under 8 cm of path wide", width < 0.08, f"{width * 100:.1f} cm", "< 8 cm")
    check("... and is quiet again 30 cm away (under 2 % of the peak)",
          p.size[(p.extra_m > 0.3) & (p.extra_m < 1.5)].max() < 0.02, None, None)
    check("the noise level is measured before the direct sound", p.noise < 0.02, p.noise, "< 0.02")
    check("steady chirps read as steady", p.steadiness > 0.95, p.steadiness, "> 0.95")

    pumping = heard(0.03, 2.0, 0.1)
    starts = [int((chirp.LEAD_S + s + 0.03) * rate) for s in chirp.CHIRP_STARTS_S]
    pumping[starts[2] - 200:] *= 0.4   # an auto gain turning itself down mid-ping
    check("an auto gain that changes level mid-ping is noticed", chirp.echo_profile(pumping).steadiness < 0.7,
          chirp.echo_profile(pumping).steadiness, "< 0.7")
    try:
        chirp.echo_profile(np.zeros(1000))
        check("a recording too short for a ping is refused", False, "accepted", "ValueError")
    except ValueError:
        check("a recording too short for a ping is refused", True)


# ------------------------------------------------------------------ ears
BOX = (4.0, 3.0)


def box_truth(x: float, y: float) -> dict[str, tuple[float, float]]:
    """The four walls of the BOX room from (x, y): name -> (room bearing deg, distance)."""
    return {"ahead (x=4)": (0.0, BOX[0] - x), "left (y=3)": (90.0, BOX[1] - y), "behind (x=0)": (180.0, x), "right (y=0)": (-90.0, y)}


def look_from(mode: str, room: SimRoom, x: float, y: float, heading_deg: float, turns=None) -> E.Look:
    g = geometry(mode)
    ears = E.Ears(g)
    turns = turns if turns is not None else (N.LOOK_TURNS_DEG if g.needs_turning else (0.0,))
    pings = []
    for turn in turns:
        heading = math.radians(heading_deg + turn)
        pings.append((heading, ears.listen(room.record((x, y, heading), g.speaker, g.mics))))
    return ears.locate(pings)


def test_ears() -> None:
    print("ears: triangulating the walls of a 4 x 3 m room from one spot")
    for mode in ("two", "four"):
        asked = found = 0
        for x, y, heading in ((1.4, 1.1, 0.0), (2.2, 1.7, 40.0), (1.8, 2.0, -120.0), (2.9, 0.8, 75.0), (0.8, 1.25, 200.0)):
            look = look_from(mode, SimRoom.box(*BOX, seed=3), x, y, heading)
            worst_m, worst_deg, missing = 0.0, 0.0, []
            for name, (bearing, distance) in box_truth(x, y).items():
                if abs(distance + 0.1 - 2.51) < 0.2:
                    continue   # the band the ceiling's echo hides (see below)
                echo = find(look.echoes, bearing, distance)
                if echo is None or abs(echo.distance_m - distance) > 0.05:
                    missing.append(name)
                else:
                    worst_m = max(worst_m, abs(echo.distance_m - distance))
                    worst_deg = max(worst_deg, off_by(echo, bearing))
            walls = sum(abs(d + 0.1 - 2.51) >= 0.2 for _, d in box_truth(x, y).values())
            asked, found = asked + walls, found + walls - len(missing)
            # One look can lose a wall (a stronger echo at the same distance, a
            # bounce off the floor cancelling it): the map is built from many looks.
            check(f"{mode} ears at ({x}, {y}) facing {heading:+.0f}: at most one wall missed", len(missing) <= 1, f"missing: {missing}", "at most one")
            check(f"  ... the walls found: distances within 3 cm, directions within 10 deg",
                  worst_m <= 0.03 and worst_deg <= 10.0, f"{worst_m * 100:.1f} cm, {worst_deg:.0f} deg", "<= 3 cm, <= 10 deg")
            behind_walls = [str(e) for e in look.echoes
                            if not (-0.25 < x + e.foot[0] < BOX[0] + 0.25 and -0.25 < y + e.foot[1] < BOX[1] + 0.25)]
            check("  ... and nothing reported beyond the walls (double bounces thrown out)",
                  len(behind_walls) <= 1, behind_walls, "at most one stray")

        check(f"{mode} ears, five spots: at least 17 walls in 20 found on a single look", found >= 0.85 * asked, f"{found} of {asked}", ">= 85 %")

    print("ears: what two ears cannot do alone, and what four can")
    room = SimRoom.box(*BOX, clutter=0.0, seed=1)
    one_heading = look_from("two", room, 3.0, 1.5, 0.0, turns=(0.0,))
    ahead, behind = find(one_heading.echoes, 0.0, 1.0), find(one_heading.echoes, 180.0, 1.0)
    check("two ears, one heading: the wall 1 m ahead is heard ...", ahead is not None and abs(ahead.distance_m - 1.0) < 0.03, ahead, "1.00 m at 0")
    check("  ... and so is its mirror twin, 1 m behind, where there is nothing",
          behind is not None and abs(behind.distance_m - 1.0) < 0.12, behind, "about 1.0 m at 180")
    turned = look_from("two", room, 3.0, 1.5, 0.0)
    check("two ears, three headings: the twin is gone", find(turned.echoes, 180.0, 1.0) is None or
          abs(find(turned.echoes, 180.0, 1.0).distance_m - 1.0) > 0.3, find(turned.echoes, 180.0, 1.0), "nothing 1 m behind")
    four = look_from("four", room, 3.0, 1.5, 0.0)
    check("four ears, one ping: no twin either", find(four.echoes, 180.0, 1.0) is None or
          abs(find(four.echoes, 180.0, 1.0).distance_m - 1.0) > 0.3, find(four.echoes, 180.0, 1.0), "nothing 1 m behind")
    check("four ears need no turning; two do", not geometry("four").needs_turning and geometry("two").needs_turning)

    print("ears: the ceiling is not a wall")
    for mode in ("two", "four"):
        hall = SimRoom.box(8.0, 8.0, clutter=0.0, ceiling_m=2.7, seed=1)   # walls too far to hear: only the ceiling answers
        look = look_from(mode, hall, 4.0, 4.0, 0.0)
        check(f"{mode} ears in a big hall under a 2.7 m ceiling hear no wall at all", not look.echoes, [str(e) for e in look.echoes], "no echoes")

    print("ears: a post, and a recording gone wrong")
    yard = SimRoom([], posts=[Post((1.0, 0.6), 0.06)], clutter=0.0, ceiling_m=None, seed=1)
    look = look_from("four", yard, 0.0, 0.0, 0.0)
    want_bearing, want_distance = math.degrees(math.atan2(0.6, 1.0)), math.hypot(1.0, 0.6) - 0.06
    post = find(look.echoes, want_bearing, want_distance, within_deg=20.0)
    check("a 12 cm post 1.2 m away, out of doors, is heard where it stands", post is not None and abs(post.distance_m - want_distance) < 0.08,
          post, f"{want_distance:.2f} m at {want_bearing:+.0f} deg")
    g = geometry("two")
    recording = SimRoom.box(*BOX, seed=1).record((1.5, 1.5, 0.0), g.speaker, g.mics)
    mono = np.stack([recording[:, 0], recording[:, 0]], axis=1)
    check("a receiver left in Mono is called out", any("Mono" in line for line in E.Ears(g).listen(mono).problems), E.Ears(g).listen(mono).problems, "a line about Mono")
    check("a clean recording has no complaints", not E.Ears(g).listen(recording).problems, E.Ears(g).listen(recording).problems, [])
    try:
        E.Ears(geometry("four")).listen(recording)
        check("two channels for four ears is refused, in words", False, "accepted", "ValueError")
    except ValueError as error:
        check("two channels for four ears is refused, in words", "4 channels" in str(error), str(error), "... needs 4 channels ...")


# ------------------------------------------------------------------ the map
def echo_of(spot, wall_toward_deg: float, wall_offset: float, strength: float = 0.7, bearing_error_deg: float = 0.0) -> E.Echo:
    """What a perfect ear at `spot` hears of the wall {p: toward . p = offset}."""
    toward = M.unit(wall_toward_deg)
    return E.Echo(wall_toward_deg + bearing_error_deg, float(wall_offset - np.dot(toward, spot)), strength, 10.0)


def test_map() -> None:
    print("map: walls, corners, and where Pupper is")
    walls = {"ahead": (0.0, 2.0), "left": (90.0, 1.5), "behind": (180.0, 1.0), "right": (-90.0, 1.0)}   # a 3 x 2.5 m room around the origin

    def hear_all(spot, error=0.0):
        return [echo_of(np.array(spot), deg, off, bearing_error_deg=error) for deg, off in walls.values()]

    room = M.RoomMap()
    room.add(np.array([0.0, 0.0]), hear_all((0, 0)))
    check("one look: four landmarks, none confirmed yet", len(room.landmarks) == 4 and not any(m.confirmed for m in room.landmarks),
          [(m.confirmed) for m in room.landmarks], "4 unconfirmed")
    room.add(np.array([0.5, 0.2]), hear_all((0.5, 0.2)))
    check("a second look from another spot: the same four, all confirmed", len(room.landmarks) == 4 and all(m.confirmed for m in room.landmarks),
          len(room.landmarks), 4)
    check("two looks are not yet enough to draw the room", len(room.outline(np.array([0.5, 0.2]))) == 4
          and abs(area(room.outline(np.array([0.5, 0.2]))) - 144.0) < 1.0, None, None)
    room.add(np.array([0.3, -0.4]), hear_all((0.3, -0.4)))
    outline = room.outline(np.array([0.5, 0.2]))
    xs, ys = [p[0] for p in outline], [p[1] for p in outline]
    check("after a third look the outline is the room: x from -1 to 2, y from -1 to 1.5",
          len(outline) == 4 and np.allclose([min(xs), max(xs), min(ys), max(ys)], [-1, 2, -1, 1.5], atol=0.02),
          [tuple(round(v, 2) for v in p) for p in outline], "a 3 x 2.5 rectangle")

    # Pupper thinks it walked 0.5 m more along x; really it went 0.42.
    truth, thought = np.array([0.92, 0.2]), np.array([1.0, 0.2])
    fixed, matched = room.fix(thought, hear_all(truth), step=thought - np.array([0.5, 0.2]))
    check("a step that fell 8 cm short is corrected by the echoes", np.linalg.norm(fixed - truth) < 0.02 and matched == 4,
          (fixed.round(3).tolist(), matched), (truth.tolist(), 4))
    fixed, matched = room.fix(thought, [], step=np.array([0.5, 0.0]))
    check("no echoes: the position stays where the steps put it", np.allclose(fixed, thought) and matched == 0, fixed.tolist(), thought.tolist())
    stray = E.Echo(45.0, 0.9, 0.9, 10.0)   # matches nothing on the map
    fixed, matched = room.fix(thought, hear_all(truth) + [stray], step=np.array([0.5, 0.0]))
    check("an echo that fits no landmark does not disturb the fix", np.linalg.norm(fixed - truth) < 0.02, fixed.round(3).tolist(), truth.tolist())

    print("map: bearings are poor, distances are good")
    tilted = M.RoomMap()   # every bearing 12 degrees off, the way two ears a hand apart can be
    spots = [np.array([0.5 * i, 0.0]) for i in range(5)]   # walking along the right-hand wall
    for spot in spots:
        tilted.add(spot, [echo_of(spot, -90.0, 1.0, bearing_error_deg=12.0)])
    wall = tilted.landmarks[0]
    facing = math.degrees(math.atan2(wall.toward[1], wall.toward[0]))
    check("a wall heard 12 deg askew from five spots is straightened by its distances",
          len(tilted.landmarks) == 1 and abs(N.wrap(facing + 90.0)) < 3.0, f"{len(tilted.landmarks)} landmark(s), facing {facing:+.1f}", "1 landmark facing -90")
    bent = M.RoomMap()
    bent.add(spots[0], [echo_of(spots[0], -90.0, 1.0, bearing_error_deg=12.0)])
    bent.add(spots[1], [echo_of(spots[1], -90.0, 1.0, bearing_error_deg=12.0)])
    fixed, _ = bent.fix(spots[2], [echo_of(spots[2], -90.0, 1.0, bearing_error_deg=12.0)], step=np.array([0.5, 0.0]))
    check("walking along a wall that sounds 12 deg askew stays straight (under 3 cm sideways)",
          abs(fixed[1]) < 0.03, f"{fixed[1] * 100:+.1f} cm sideways", "|sideways| < 3 cm")

    corner_at = np.array([2.0, 1.5])
    corner = M.RoomMap()
    for spot in ([0.0, 0.0], [0.6, -0.3], [1.0, 0.6], [0.2, 0.8]):
        spot = np.array(spot)
        to = corner_at - spot
        corner.add(spot, [E.Echo(math.degrees(math.atan2(to[1], to[0])) + 6.0, float(np.linalg.norm(to)), 0.6, 10.0)])
    mark = corner.landmarks[0]
    check("something that answers from one point, wherever Pupper stands, is a corner",
          len(corner.landmarks) == 1 and mark.is_corner and np.linalg.norm(mark.point - corner_at) < 0.1,
          (len(corner.landmarks), mark.is_corner, mark.point.round(2).tolist()), (1, True, corner_at.tolist()))
    check("a wall is not mistaken for one", not wall.is_corner)

    print("map: how far can Pupper go?")
    here = np.array([0.5, 0.2])
    look = hear_all(here)
    check("straight at the wall ahead: its distance", abs(room.clearance(here, 0.0, look) - 1.5) < 0.01, room.clearance(here, 0.0, look), 1.5)
    check("at 45 deg, between the walls ahead and to the left: the nearer crossing",
          abs(room.clearance(here, 45.0, look) - min(1.5, 1.3) / math.cos(math.radians(45))) < 0.02, room.clearance(here, 45.0, look), 1.84)
    weak = [E.Echo(0.0, 1.0, 0.25, 10.0)]   # a chair leg dead ahead
    empty = M.RoomMap()
    check("a weak echo dead ahead blocks the way ahead ...", abs(empty.clearance(here, 0.0, weak) - 1.0) < 0.01, empty.clearance(here, 0.0, weak), 1.0)
    check("  ... but not the way past it, 45 deg to the side", empty.clearance(here, 45.0, weak) > 5.0, empty.clearance(here, 45.0, weak), "far")
    solid = [E.Echo(0.0, 1.0, 0.8, 10.0)]
    check("a solid echo is taken for a wall that runs on: 45 deg to the side is blocked too",
          abs(empty.clearance(here, 45.0, solid) - 1.0 / math.cos(math.radians(45))) < 0.02, empty.clearance(here, 45.0, solid), 1.41)
    json.dumps(room.as_dict())
    check("the map can be sent to the page (JSON)", True)


# ------------------------------------------------------------------ the navigator
def make_navigator(mode: str, room: SimRoom, start: tuple[float, float, float], seed: int = 0):
    g = geometry(mode)
    dog = SimDog(*start, seed=seed)
    dog.standing = True
    # The pretend dog's speaker and ears are up to a centimetre from where ears.yaml says.
    nudge = np.random.default_rng(seed + 10)
    flat = [1.0, 1.0, 0.0]
    sound = S.SimSound(room, dog, g.speaker + 0.007 * nudge.standard_normal(3) * flat,
                       g.mics + 0.007 * nudge.standard_normal(g.mics.shape) * flat)
    lines: list[str] = []
    return N.Navigator(dog, sound, E.Ears(g), say=lines.append, background=False), dog, lines


def run(nav: N.Navigator, dog: SimDog, room: SimRoom, limit_s: float = 400.0, until=None) -> tuple[float, float]:
    """Tick at 20 Hz on a pretend clock until the job ends. Returns (how long, how close the dog came to anything)."""
    now, closest = 0.0, room.distance_to_wall((dog.x, dog.y))
    while nav.busy and now < limit_s and not (until and until(now)):
        now += 0.05
        nav.tick(now)
        closest = min(closest, room.distance_to_wall((dog.x, dog.y)))
    return now, closest


def to_room(start: tuple[float, float, float], p) -> np.ndarray:
    """A point in the navigator's frame (origin: the start pose) -> the pretend room's frame."""
    c, s = math.cos(math.radians(start[2])), math.sin(math.radians(start[2]))
    return np.array([start[0] + c * p[0] - s * p[1], start[1] + s * p[0] + c * p[1]])


def to_map(start: tuple[float, float, float], p) -> tuple[float, float]:
    c, s = math.cos(math.radians(start[2])), math.sin(math.radians(start[2]))
    dx, dy = p[0] - start[0], p[1] - start[1]
    return (c * dx + s * dy, -s * dx + c * dy)


def area(shape: list) -> float:
    return 0.5 * abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(shape, shape[1:] + shape[:1])))


def test_navigator() -> None:
    print("navigator: exploring a 4 x 3 m room blind (the dog walks 15 % short, its gyro drifts)")
    start = (1.2, 1.0, 20.0)
    for mode in ("two", "four"):
        room = SimRoom.box(*BOX, seed=3)
        nav, dog, lines = make_navigator(mode, room, start)
        check(f"{mode} ears: explore starts", nav.start("explore", 0.0) == "", None, None)
        seconds, closest = run(nav, dog, room)
        check(f"  finished by itself ({seconds:.0f} s on the pretend clock, {dog.travelled:.1f} m walked)", not nav.busy and "explored" in nav.message, nav.message, "explored: ...")
        check("  never nearer than 0.25 m to a wall", closest >= 0.25, f"{closest:.2f} m", ">= 0.25 m")
        error = float(np.linalg.norm(to_room(start, (nav.x, nav.y)) - [dog.x, dog.y]))
        check("  knows where it is to 15 cm, after 3 m of walking short", error < 0.15, f"{error * 100:.0f} cm", "< 15 cm")
        outline = [to_room(start, p) for p in nav.map.outline(np.array([nav.x, nav.y]))]
        size = area(outline)
        corners_off = max((min(np.linalg.norm(p - c) for c in ([0, 0], [4, 0], [4, 3], [0, 3])) for p in outline), default=9.0)
        check("  the room it drew is the room: 12 m2 within 10 %, corners within 20 cm",
              abs(size - 12.0) < 1.2 and corners_off < 0.20 and len(outline) == 4,
              f"{size:.1f} m2, {len(outline)} corners, worst {corners_off * 100:.0f} cm off", "12 m2, 4 corners, < 20 cm")
        confirmed = sum(m.confirmed and not m.is_corner for m in nav.map.landmarks)
        check("  four walls confirmed (a stray fifth allowed)", 4 <= confirmed <= 5, confirmed, "4 or 5")

    print("navigator: walking to a point")
    for mode, room_maker, start, goal in (
            ("two", lambda: SimRoom.box(*BOX, seed=5), (1.2, 1.0, 20.0), (3.0, 2.2)),
            ("four", lambda: SimRoom.polygon([(0, 0), (4.2, 0), (3.4, 3.0), (0, 2.6)], seed=5), (1.0, 1.0, 70.0), (2.6, 2.0)),
            ("four", lambda: SimRoom(SimRoom.box(4.5, 3.5).walls, posts=[Post((2.4, 1.5), 0.06)], seed=5), (1.1, 1.3, 10.0), (3.6, 1.7))):
        room = room_maker()
        nav, dog, lines = make_navigator(mode, room, start, seed=1)
        nav.start("goto", 0.0, to_map(start, goal))
        seconds, closest = run(nav, dog, room)
        missed = math.hypot(dog.x - goal[0], dog.y - goal[1])
        what = "past a post" if room.posts else ("in a slanted room" if len(room.walls) == 4 and room.walls[1].b[0] != room.walls[1].a[0] else "across the room")
        check(f"{mode} ears, {what}: arrives", "arrived" in nav.message, nav.message, "arrived ...")
        check(f"  really within 30 cm of the goal, never nearer than 0.2 m to anything",
              missed < 0.30 and closest >= 0.2, f"{missed * 100:.0f} cm from the goal, closest {closest:.2f} m", "< 30 cm, >= 0.2 m")

    print("navigator: saying no, and stopping")
    room = SimRoom.box(*BOX, seed=3)
    nav, dog, lines = make_navigator("two", room, start)
    dog.standing = False
    check("two ears, lying down: a look is refused (it needs to turn)", "lying down" in nav.start("look", 0.0), nav.start("look", 0.0), "... lying down")
    nav4, dog4, _ = make_navigator("four", room, start)
    dog4.standing = False
    check("four ears, lying down: a look is fine (nothing moves)", nav4.start("look", 0.0) == "")
    run(nav4, dog4, room)
    check("  ... and hears the room", len(nav4.echoes) >= 4 and dog4.travelled == 0.0, len(nav4.echoes), ">= 4 echoes, 0 m walked")
    check("  ... but exploring is refused", "lying down" in nav4.start("explore", 1.0))
    dog.standing = True
    nav.start("explore", 0.0)
    run(nav, dog, room, until=lambda now: nav.state == N.State.WALK)
    dog.standing = False   # the e-stop, or a fall
    nav.tick(99.0)
    check("the e-stop mid-step ends the job", not nav.busy and "stopped" in nav.message, nav.message, "stopped: ...")
    dog.standing = True
    nav.start("explore", 100.0)
    nav.stop("stopped by hand")
    check("stop() ends the job at once", not nav.busy and nav.job == "")

    class Deaf:
        name = "broken"

        def ping(self):
            raise OSError("no such device")

    nav.sound = Deaf()
    nav.start("look", 200.0)
    now = 200.0
    while nav.busy and now < 230.0:
        now += 0.05
        nav.tick(now)
    check("a sound card that fails stops the job and says why", "could not listen" in nav.message and "no such device" in nav.message, nav.message, "... could not listen (no such device)")
    json.dumps(nav4.snapshot())
    check("the report for the map page is JSON", True)

    cramped = SimRoom.box(0.9, 0.9, clutter=0.0, ceiling_m=None, seed=1)
    nav, dog, _ = make_navigator("four", cramped, (0.45, 0.45, 0.0))
    nav.start("explore", 0.0)
    run(nav, dog, cramped)
    check("in a 0.9 m box the echoes pile up: it says it is too tight and stays put",
          "too tight" in nav.message and dog.travelled < 0.05, (nav.message, round(dog.travelled, 2)), "too tight ..., 0 m")
    snug = SimRoom.box(1.3, 1.2, clutter=0.0, ceiling_m=None, seed=1)
    nav, dog, _ = make_navigator("four", snug, (0.7, 0.55, 0.0))
    nav.start("explore", 0.0)
    _, closest = run(nav, dog, snug)
    check("in a 1.3 x 1.2 m box it may shuffle, but stays 0.2 m off the walls", closest >= 0.2, f"{closest:.2f} m ({nav.message})", ">= 0.2 m")


# ------------------------------------------------------------------ the tools
def test_tools() -> None:
    print("tools: mic_check.py, the sound card, the server")
    check("mic_check --sim finds nothing wrong", mic_check.main(["--sim", "--ears", "four", "--pings", "2"]) == 0)

    # --wall: start from direct paths that are 2.5 cm too short, as if measured through the body.
    g = geometry("two")
    g.direct_m = g.direct_m - 0.025
    with tempfile.TemporaryDirectory() as folder:
        kept, E.CALIBRATION_FILE = E.CALIBRATION_FILE, Path(folder) / "calibration.json"
        try:
            room = SimRoom.box(*BOX, clutter=0.3, floor_reflect=0.0, seed=2)   # no floor bounce riding on the wall's echo
            true = geometry("two")
            sound = S.SimSound(room, mic_check.StillDog(3.0, 1.5, 0.0), true.speaker, true.mics)
            mic_check.report(g, sound, 3, 1.0, None)
            saved = json.loads(E.CALIBRATION_FILE.read_text())["two"]["direct_m"]
            again = E.load_geometry("two", calibration=E.CALIBRATION_FILE)
        finally:
            E.CALIBRATION_FILE = kept
    check("--wall 1.0 recovers direct paths that were 2.5 cm off (to 5 mm)", np.allclose(saved, true.direct_m, atol=0.005),
          [round(v, 3) for v in saved], true.direct_m.round(3).tolist())
    check("... and ears.py loads them", np.allclose(again.direct_m, saved))

    # The real sound card, with a stand-in for the sounddevice module.
    recording = SimRoom.box(*BOX, seed=1).record((1.5, 1.5, 0.0), true.speaker, true.mics).astype(np.float32)
    asked: dict = {}

    class Stream:
        def __init__(self, **settings):
            self.settings = settings
            asked.setdefault("streams", []).append(settings)

        def __enter__(self):
            if "callback" in self.settings:   # the input stream: hand over the recording in blocks
                for i in range(0, len(recording), 1024):
                    self.settings["callback"](recording[i:i + 1024], 1024, None, None)
            return self

        def __exit__(self, *_):
            return False

        def write(self, data):
            asked["played"] = data.shape

    devices = [{"name": "Wireless Microphone RX (DJI)", "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 48000.0},
               {"name": "Pupper speaker", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000.0}]
    fake = types.SimpleNamespace(
        query_devices=lambda device=None, kind=None: devices if device is None and kind is None else devices[device if device is not None else (0 if kind == "input" else 1)],
        InputStream=Stream, OutputStream=Stream)
    kept_module, kept_extra = sys.modules.get("sounddevice"), S.EXTRA_S
    sys.modules["sounddevice"], S.EXTRA_S = fake, 0.0
    try:
        card = S.SoundCard(2, mic="dji", speaker="speaker")
        heard = card.ping()
        check("SoundCard finds the DJI receiver by name and records one channel per ear",
              card.mic == 0 and heard.shape == recording.shape and asked["streams"][0]["channels"] == 2, (card.mic, heard.shape), (0, recording.shape))
        check("... at 48 kHz, playing the whole ping", asked["streams"][0]["samplerate"] == 48000 and asked["played"][0] == len(chirp.make_ping()))
        check("... and what it recorded makes sense to the ears", len(E.Ears(true).hear(heard).echoes) >= 3)
        try:
            S.SoundCard(4, mic="dji")
            check("four ears on a two-channel receiver is refused, with what to do", False, "accepted", "ValueError")
        except ValueError as error:
            check("four ears on a two-channel receiver is refused, with what to do", "Quadraphonic" in str(error), str(error), "... Quadraphonic ...")
    finally:
        S.EXTRA_S = kept_extra
        if kept_module is None:
            del sys.modules["sounddevice"]
        else:
            sys.modules["sounddevice"] = kept_module

    # The server's session, without the web part: a four-ear look on the real clock.
    import server

    async def session_look() -> dict:
        session = server.Session(argparse.Namespace(sim=True, ears="four", room="arena", mic=None, speaker=None))
        await session.do({"do": "stand"})
        await session.do({"do": "look"})
        task = asyncio.create_task(session.act())
        for _ in range(100):
            await asyncio.sleep(0.05)
            session.page_heard = asyncio.get_event_loop().time() + 1e9   # the page keeps saying hello
            if not session.navigator.busy:
                break
        task.cancel()
        return session.report()

    report = asyncio.run(session_look())
    json.dumps(report)
    check("server: stand up, look -> a report with echoes, the map and the real room (sim)",
          len(report["echoes"]) >= 4 and report["truth"]["walls"] and "looked" in report["message"], report["message"], "looked: n echoes")


PARTS = {"chirp": test_chirp, "ears": test_ears, "map": test_map, "navigator": test_navigator, "tools": test_tools}

if __name__ == "__main__":
    for part in (sys.argv[1:] or PARTS):
        if part not in PARTS:
            sys.exit(f"unknown part '{part}': choose from {', '.join(PARTS)}")
        PARTS[part]()
    print(f"\n{COUNT[0] - len(FAILED)} of {COUNT[0]} checks passed" + ("" if not FAILED else f"; FAILED: {len(FAILED)}"))
    for name in FAILED:
        print(f"  FAIL  {name}")
    sys.exit(1 if FAILED else 0)
