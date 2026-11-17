"""Desk-side tests: no robot, no ROS, no browser.

    python3 tests/test.py              # everything
    python3 tests/test.py perception   # one part: perception, policy, moves, squat, camera

Like the labs' test.py: every check prints what it wanted and what it got.
The pictures are drawn here with the game's own projection, plus a few real
frames of the game (tests/fixtures) with the truth saved next to them.
The camera tests build a pretend fisheye picture of a screen showing the game,
from a known spot on the floor, and check that Pupper finds both the game and
the spot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "surfer"))

import moves  # noqa: E402
import perception as P  # noqa: E402
import policy  # noqa: E402
import screen_finder as S  # noqa: E402
import squat_reference  # noqa: E402
from robot_io import SimRobot, load_reference  # noqa: E402

FAILED = []
COUNT = [0]


def check(name: str, ok: bool, got=None, want=None) -> None:
    COUNT[0] += 1
    if ok:
        print(f"  ok    {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}\n          got:    {got}\n          wanted: {want}")


# ------------------------------------------------------------------ drawing
PALETTE = {k: P.hex_to_bgr(v) for k, v in P.CONFIG["palette"].items()}
OBSTACLES = P.CONFIG["obstacles"]


def draw_view(rows: list[tuple[float, list[str]]], runner_lane: int = 1, width=P.VIEW_W, height=P.VIEW_H) -> np.ndarray:
    """A plain version of the game view: the track, obstacles as flat shapes, the runner."""
    view = np.zeros((height, width, 3), np.uint8)
    view[:] = PALETTE["sky_top"]
    track = np.array([[P.lane_x(-0.5, 500) * width, P.y_of(500) * height], [P.lane_x(2.5, 500) * width, P.y_of(500) * height],
                      [P.lane_x(2.5, -4) * width, P.y_of(-4) * height], [P.lane_x(-0.5, -4) * width, P.y_of(-4) * height]])
    cv2.fillPoly(view, [track.astype(np.int32)], PALETTE["track"])

    def rect(lane, z, half, bottom, top, color):
        unit = P.LANE_WIDTH * width * P.scale_at(z)
        x, y = P.lane_x(lane, z) * width, P.y_of(z) * height
        cv2.rectangle(view, (int(round(x - half * unit)), int(round(y - top * unit))),
                      (int(round(x + half * unit)) - 1, int(round(y - bottom * unit)) - 1), color, -1)

    for z, cells in sorted(rows, reverse=True):
        for lane, cell in enumerate(cells):
            if cell == P.WALL:
                rect(lane, z, OBSTACLES["wall"]["half_width"], 0, OBSTACLES["wall"]["height"], PALETTE["wall"])
            elif cell == P.HURDLE:
                rect(lane, z, OBSTACLES["hurdle"]["half_width"], 0, OBSTACLES["hurdle"]["height"], PALETTE["hurdle"])
            elif cell == P.BEAM:
                beam = OBSTACLES["beam"]
                stripe = np.array([[P.lane_x(lane - beam["half_width"], z), P.y_of(z)], [P.lane_x(lane + beam["half_width"], z), P.y_of(z)],
                                   [P.lane_x(lane + beam["half_width"], z + beam["ground_stripe"]), P.y_of(z + beam["ground_stripe"])],
                                   [P.lane_x(lane - beam["half_width"], z + beam["ground_stripe"]), P.y_of(z + beam["ground_stripe"])]])
                cv2.fillPoly(view, [(stripe * [width, height]).astype(np.int32)], PALETTE["beam"])
                rect(lane, z, beam["half_width"], beam["bottom"], beam["top"], PALETTE["beam"])
    rect(runner_lane, 0, 0.25, 0, 0.8, PALETTE["dog"])
    return view


def cells_seen(hazards: list[P.Hazard], z: float, tolerance: float = 0.12) -> list[str]:
    cells = [P.FREE] * 3
    for h in hazards:
        if abs(h.z - z) <= max(1.5, tolerance * z):
            cells[h.lane] = h.kind
    return cells


# ------------------------------------------------------------------ perception
def test_perception() -> None:
    print("perception: one picture -> obstacles")
    errors = [abs(P.z_from_y(P.y_of(z)) - z) for z in (0, 3, 10, 25, 60)]
    check("z_from_y undoes y_of", max(errors) < 1e-6, max(errors), "0")

    hues = sorted([*P.HUES.values(), P.BORDER_HUE])
    gaps = [b - a for a, b in zip(hues, hues[1:])] + [hues[0] + 180 - hues[-1]]
    check("the obstacle colors and the frame's green are far apart in hue",
          min(gaps) > 2 * P.HUE_TOLERANCE, f"smallest gap {min(gaps)}", f"more than {2 * P.HUE_TOLERANCE}")

    scene = [(12.0, [P.HURDLE, P.BEAM, P.FREE]), (30.0, [P.FREE, P.FREE, P.WALL])]
    hazards = P.detect(draw_view(scene))
    check("a hurdle, a beam and a wall are found in the right lanes",
          cells_seen(hazards, 12.0) == scene[0][1] and cells_seen(hazards, 30.0) == scene[1][1],
          [(h.lane, h.kind, round(h.z, 1)) for h in hazards], scene)
    near = [h for h in hazards if h.z < 20]
    worst = max((abs(h.z - 12.0) for h in near), default=99)
    check("the bottom edge gives the distance (12 units, within 0.6)", worst < 0.6, f"off by {worst:.2f}", "< 0.6")
    beams = [h for h in hazards if h.kind == P.BEAM]
    check("a beam is one obstacle, not its stripe and its bar", len(beams) == 1, len(beams), 1)

    check("an empty track has no obstacles (the runner is not one)", P.detect(draw_view([])) == [], P.detect(draw_view([])), [])

    behind = P.detect(draw_view([(8.0, [P.HURDLE, P.FREE, P.FREE]), (10.0, [P.WALL, P.FREE, P.FREE])]))
    wall = [h for h in behind if h.kind == P.WALL]
    check("an obstacle whose bottom edge is hidden by a nearer one is marked not exact",
          len(wall) == 1 and not wall[0].exact, [(h.kind, h.exact) for h in behind], "the wall: exact=False")

    wrong = []
    for truth_path in sorted((ROOT / "tests" / "fixtures").glob("*.json")):
        truth = json.loads(truth_path.read_text())
        hazards = P.detect(cv2.imread(str(truth_path.with_suffix(".png"))))
        for row in truth["rows"]:
            if not 4.5 < row["z"] < 40:
                continue  # nearer: the runner hides it; farther: a few pixels
            # A wall hides the rows behind it in its own lanes: only check lanes in clear view.
            hidden = {lane for other in truth["rows"] if other["z"] < row["z"]
                      for lane, cell in enumerate(other["cells"]) if cell == P.WALL and row["z"] - other["z"] < 30}
            seen = cells_seen(hazards, row["z"])
            for lane in set(range(3)) - hidden:
                if seen[lane] != row["cells"][lane]:
                    wrong.append((truth_path.stem, row["z"], lane, seen[lane], row["cells"][lane]))
    check("real frames of the game: every row in clear view is read correctly", not wrong, wrong[:4], "no mistakes")


def test_tracker() -> None:
    print("perception: following rows over time")
    tracker = P.RowTracker(nominal_speed=6.0)  # a wrong first guess: it must measure the real 8
    speed, z0, dt = 8.0, 40.0, 1 / 15
    seen_near = None
    for i in range(int(5.05 / dt)):
        now = i * dt
        z = z0 - speed * now
        # Nearer than 4 units the runner hides the row: no detections at all.
        view = draw_view([(z, [P.HURDLE, P.FREE, P.WALL])] if z > 4.0 else [])
        rows = tracker.update(P.detect(view), now)
        if 0.5 < z < 1.0:
            seen_near = (z, [(round(r.z, 2), r.cells) for r in rows])
    check("the track speed is measured (8 units/s, within 10%)", abs(tracker.speed - speed) < 0.8, round(tracker.speed, 2), "8.0")
    ok = seen_near is not None and len(seen_near[1]) == 1 and abs(seen_near[1][0][0] - seen_near[0]) < 0.8
    check("a row hidden behind the runner is carried forward to where it must be", ok, seen_near, "one row within 0.8 of the truth")
    check("... and keeps what was in its lanes", seen_near and seen_near[1][0][1] == [P.HURDLE, P.FREE, P.WALL],
          seen_near and seen_near[1][0][1], [P.HURDLE, P.FREE, P.WALL])
    tracker.update([], 5.5)
    check("a row behind the runner is forgotten", tracker.rows == [], [(r.z, r.cells) for r in tracker.rows], [])


# ------------------------------------------------------------------ policy
def row(z: float, cells: list[str], handled: bool = False) -> P.Row:
    r = P.Row(z=z)
    for lane, cell in enumerate(cells):
        if cell != P.FREE:
            r.votes[lane][cell] = 3
    r.handled = handled
    return r


def test_policy() -> None:
    print("policy: rows -> a move")
    T = policy.Timing(shuffle_s=0.8, jump_lead_s=0.72, squat_lead_s=0.6)
    F, H, B, W = P.FREE, P.HURDLE, P.BEAM, P.WALL
    v = 8.0

    def move(rows, lane):
        return policy.decide(rows, lane, v, T)[0]

    check("nothing ahead: keep running", move([], 1) == "run", move([], 1), "run")
    check("a free lane ahead: keep running", move([row(20, [W, F, W])], 1) == "run", move([row(20, [W, F, W])], 1), "run")
    check("a hurdle still 2 s away: wait", move([row(16, [F, H, F])], 1) == "run", move([row(16, [F, H, F])], 1), "run")
    check("a hurdle 0.7 s away: jump", move([row(5.6, [F, H, F])], 1) == "jump", move([row(5.6, [F, H, F])], 1), "jump")
    check("a beam 0.55 s away: squat", move([row(4.4, [B, B, B])], 1) == "squat", move([row(4.4, [B, B, B])], 1), "squat")
    check("a hurdle it already jumped for: no second jump",
          move([row(3.0, [H, H, H], handled=True)], 1) == "run", move([row(3.0, [H, H, H], handled=True)], 1), "run")
    check("a wall ahead, the left lane free: shuffle left, at once (not at the last moment)",
          move([row(22, [F, W, W])], 1) == "left", move([row(22, [F, W, W])], 1), "left")
    check("a wall ahead, the right lane free: shuffle right", move([row(22, [W, W, F])], 1) == "right", move([row(22, [W, W, F])], 1), "right")
    check("walls on both sides, a hurdle in its lane: never shuffle into a wall",
          move([row(20, [W, H, W])], 1) == "run" and move([row(5, [W, H, W])], 1) == "jump",
          (move([row(20, [W, H, W])], 1), move([row(5, [W, H, W])], 1)), ("run", "jump"))
    got = move([row(20, [F, W, F]), row(44, [W, W, F])], 1)
    check("both sides free now: pick the side that works for the row after", got == "right", got, "right")
    got = move([row(20, [F, F, H]), row(44, [W, W, F])], 2)
    check("in the far-left lane's way later, a hurdle now: stay and jump, don't wander", got == "run", got, "run")
    got = move([row(1.6, [F, H, W])], 2)
    check("a wall 0.2 s away, too late to do it properly: shuffle anyway (a wall costs more than being late)",
          got == "left", got, "left (the only chance)")
    got = move([row(4.0, [H, F, F])], 1)
    check("a free lane: don't shuffle into a hurdle lane", got == "run", got, "run")
    check("a row behind the runner is ignored", move([row(-1.0, [W, W, W])], 1) == "run", move([row(-1.0, [W, W, W])], 1), "run")


# ------------------------------------------------------------------ moves
class Clock:
    def __init__(self) -> None:
        self.now = 100.0


def run_surfer(surfer: moves.Surfer, robot: SimRobot, clock: Clock, seconds: float, position=None) -> None:
    for _ in range(int(round(seconds * 20))):
        before = len(robot.commands)
        surfer.tick(clock.now, position)
        if len(robot.commands) > before:  # stamp the command with the test's clock
            robot.commands[-1] = (clock.now, *robot.commands[-1][1:])
        clock.now += 0.05


def test_moves() -> None:
    print("moves: the five moves as speed commands and tables")
    robot, sent, clock = SimRobot(), [], Clock()
    surfer = moves.Surfer(robot, sent.append)

    run_surfer(surfer, robot, clock, 0.5)
    check("IDLE says nothing (the joystick keeps working)", robot.commands == [], len(robot.commands), 0)

    surfer.start(clock.now)
    run_surfer(surfer, robot, clock, 1.6)
    vy = [c[2] for c in robot.commands]
    check("running in place: a sideways speed of at least 0.1 (the lift gait is on), no forward speed",
          all(abs(c[2]) >= 0.1 and c[1] == 0 and c[3] == 0 for c in robot.commands), sorted(set(vy)), "|vy| >= 0.1, vx = wz = 0")
    check("... that flips sign, and adds up to nowhere", abs(sum(vy)) * 0.05 < 0.02 and min(vy) < 0 < max(vy), round(sum(vy) * 0.05, 3), "0 m (within one tick)")

    robot.commands.clear()
    check("shuffle left is accepted", surfer.begin("left", clock.now), False, True)
    check("... the game hears about it at once, with how long it takes",
          sent[-1] == {"type": "move", "move": "left", "duration": moves.SHUFFLE_S, "lane": 0}, sent[-1], "move left, lane 0")
    check("... and nothing else can start meanwhile", not surfer.begin("jump", clock.now), True, False)
    run_surfer(surfer, robot, clock, moves.SHUFFLE_S + 0.15)
    travelled = sum(c[2] for c in robot.commands if c[2] == moves.SHUFFLE_VY) * 0.05
    check("... +vy (left) for SHUFFLE_S, then running in place again in lane 0",
          surfer.state == moves.State.RUN and surfer.lane == 0 and abs(travelled - moves.SHUFFLE_VY * moves.SHUFFLE_S) < 0.03,
          (surfer.state.name, surfer.lane, round(travelled, 2)), ("RUN", 0, moves.SHUFFLE_VY * moves.SHUFFLE_S))
    check("no lane left of the left lane", not surfer.begin("left", clock.now), True, False)

    robot.commands.clear()
    robot.references.clear()
    surfer.begin("jump", clock.now)
    started = clock.now
    run_surfer(surfer, robot, clock, 0.2)
    check("jump: stands still first, table not sent yet", robot.references == [] and all(c[1:] == (0, 0, 0) for c in robot.commands),
          (robot.references, robot.commands[-1][1:]), "no table, zero speed")
    run_surfer(surfer, robot, clock, 0.2)
    check("... then sends the jump table once, and tells the game",
          [r[1] for r in robot.references] == ["jump"] and sent[-1]["move"] == "jump" and sent[-1]["duration"] == robot.reference_seconds("jump"),
          ([r[1] for r in robot.references], sent[-1]), "one jump table")
    run_surfer(surfer, robot, clock, 3.0)
    busy = next(c[0] for c in robot.commands if c[2] != 0) - started
    check("... and leaves the policy alone until it has landed (zero speed, at least 1.0 s after the table)",
          busy - moves.JUMP_SETTLE_S >= 1.0 and [r[1] for r in robot.references] == ["jump"], round(busy - moves.JUMP_SETTLE_S, 2), ">= 1.0 s")

    surfer.begin("squat", clock.now)
    started = clock.now
    run_surfer(surfer, robot, clock, 3.0)
    squat_at = next(i for i, s in enumerate(sent) if s["move"] == "squat")
    check("squat: the squat table, once", [r[1] for r in robot.references] == ["jump", "squat"] and squat_at == len(sent) - 1,
          [r[1] for r in robot.references], ["jump", "squat"])

    timing = surfer.timing()
    check("the policy is told how early to decide: jump lead = settle + half the jump",
          abs(timing.jump_lead_s - (moves.JUMP_SETTLE_S + 0.47)) < 0.01, timing.jump_lead_s, moves.JUMP_SETTLE_S + 0.47)

    surfer.stop()
    robot.commands.clear()
    run_surfer(surfer, robot, clock, 0.3)
    check("stop: stands still and goes quiet", surfer.state == moves.State.IDLE and robot.commands == [], len(robot.commands), 0)

    # Holding the lane with the camera.
    robot, sent, clock = SimRobot(), [], Clock()
    surfer = moves.Surfer(robot, sent.append, hold=True)
    surfer.start(clock.now)
    home = moves.Position(left=0.0, distance=0.6, facing_right_deg=0.0)
    run_surfer(surfer, robot, clock, 0.1, home)
    robot.commands.clear()
    run_surfer(surfer, robot, clock, 0.1, moves.Position(left=-0.10, distance=0.6, facing_right_deg=0.0))
    check("hold: drifted 10 cm right of its spot -> steer left (+vy)", robot.commands[-1][2] > 0.1, robot.commands[-1][1:], "+vy")
    run_surfer(surfer, robot, clock, 0.1, moves.Position(left=0.0, distance=0.6, facing_right_deg=12.0))
    check("hold: turned 12 degrees right -> turn left (+wz)", robot.commands[-1][3] > 0, robot.commands[-1][1:], "+wz")
    run_surfer(surfer, robot, clock, 0.1, moves.Position(left=0.0, distance=0.75, facing_right_deg=0.0))
    check("hold: 15 cm too far from the screen -> walk forward (+vx)", robot.commands[-1][1] > 0, robot.commands[-1][1:], "+vx")
    surfer.begin("right", clock.now)
    run_surfer(surfer, robot, clock, 0.1, home)
    check("hold: shuffle right steers right (-vy) toward the lane, not by the clock", robot.commands[-1][2] < -0.2, robot.commands[-1][1:], "-vy")
    run_surfer(surfer, robot, clock, 0.6, moves.Position(left=-moves.LANE_SPACING_M, distance=0.6, facing_right_deg=0.0))
    check("... and ends when the camera says it has arrived", surfer.state == moves.State.RUN, surfer.state.name, "RUN")


def test_squat() -> None:
    print("squat: the reference table")
    ref = squat_reference.build()
    table = np.array(ref["table"])
    check("12 joint angles per row, inside the joint limits",
          table.shape[1] == 12 and np.all(table >= squat_reference.JOINT_LOW) and np.all(table <= squat_reference.JOINT_HIGH),
          table.shape, "(n, 12), in limits")
    check("starts and ends standing", np.allclose(table[0], squat_reference.STANDING, atol=1e-3) and np.allclose(table[-1], squat_reference.STANDING, atol=1e-3),
          np.round(table[0], 2).tolist(), squat_reference.STANDING.tolist())
    low = table[len(table) // 2]
    slides, drops = [], []
    for i, leg in enumerate(squat_reference.LEGS):
        a = squat_reference.foot_position(leg, squat_reference.STANDING[3 * i:3 * i + 3])
        b = squat_reference.foot_position(leg, low[3 * i:3 * i + 3])
        slides.append(float(np.linalg.norm((b - a)[:2])))
        drops.append(float(b[2] - a[2]))
    check("at the bottom every foot is DROP_M closer to the body", np.allclose(drops, squat_reference.DROP_M, atol=1e-3), np.round(drops, 4).tolist(), squat_reference.DROP_M)
    check("... and has not slid on the floor (under 1 mm)", max(slides) < 1e-3, f"{1000 * max(slides):.2f} mm", "< 1 mm")
    step = np.abs(np.diff(table, axis=0)).max() / (ref["playback_s"] / (len(table) - 1))
    check("no joint is asked to move faster than 5 rad/s", step < 5.0, f"{step:.1f} rad/s", "< 5")
    seconds, data = load_reference(squat_reference.OUT)
    check("the saved file is the message the controller wants: 3 numbers, then n x 12",
          (len(data) - 3) % 12 == 0 and data[0] == seconds >= data[1] and seconds + moves.SQUAT_RECOVER_S + moves.SQUAT_SETTLE_S >= 1.0,
          (len(data), data[:3]), "3 + n*12 values; playback >= active")


# ------------------------------------------------------------------ camera
def camera_picture(view: np.ndarray, left: float, distance: float, facing_right_deg: float, width_m: float,
                   focal: float, size: tuple[int, int], height_m: float = 0.0):
    """A pinhole camera's picture of a screen showing the game view inside its green frame.

    The camera stands `left` m left of the frame's centre line, `distance` m in
    front of the screen, turned `facing_right_deg` clockwise. Returns the
    picture and the frame's four corners in it.
    """
    border = int(round(P.CONFIG["view"]["border"] * view.shape[1] / P.CONFIG["view"]["width"]))
    framed = cv2.copyMakeBorder(view, border, border, border, border, cv2.BORDER_CONSTANT, value=PALETTE["border_pupper"])
    h, w = framed.shape[:2]
    half_w, half_h = width_m / 2, width_m / 2 * h / w
    corners_3d = np.array([[-half_w, -half_h, 0], [half_w, -half_h, 0], [half_w, half_h, 0], [-half_w, half_h, 0]])
    a = np.radians(facing_right_deg)
    rotation = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])  # rows: the camera's axes
    centre = np.array([-left, height_m, -distance])
    in_camera = (corners_3d - centre) @ rotation.T
    corners = np.stack([focal * in_camera[:, 0] / in_camera[:, 2] + size[0] / 2,
                        focal * in_camera[:, 1] / in_camera[:, 2] + size[1] / 2], axis=1).astype(np.float32)
    warp = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), corners)
    room = np.full((size[1], size[0], 3), (70, 75, 80), np.uint8)       # a grey room
    cv2.rectangle(room, (0, int(size[1] * 0.62)), (size[0], size[1]), (60, 90, 120), -1)  # a wooden floor
    picture = cv2.warpPerspective(framed, warp, size, dst=room, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.INTER_AREA)
    return picture, corners


def fisheye_picture(scene: np.ndarray, scene_focal: float, size=(1400, 1050)) -> np.ndarray:
    """What Pupper's fisheye lens makes of a pinhole picture of the scene.

    Uses the double sphere model's *unprojection* (Usenko et al., 2018, eq. 46-50):
    the opposite direction from the projection screen_finder.Pinhole uses, so
    this is an independent check of that math.
    """
    import yaml
    p = yaml.safe_load(S.CAMERA_PARAMS.read_text())["camera_params"]
    u, v = np.meshgrid(np.arange(size[0], dtype=np.float64), np.arange(size[1], dtype=np.float64))
    mx, my = (u - p["cx"]) / p["fx"], (v - p["cy"]) / p["fy"]
    r2 = mx * mx + my * my
    alpha, xi = p["alpha"], p["xi"]
    inside = 1 - (2 * alpha - 1) * r2
    mz = (1 - alpha * alpha * r2) / (alpha * np.sqrt(np.maximum(inside, 0)) + 1 - alpha)
    k = (mz * xi + np.sqrt(mz * mz + (1 - xi * xi) * r2)) / (mz * mz + r2)
    x, y, z = k * mx, k * my, k * mz - xi
    ahead = (inside > 0) & (z > 0.05)
    map_x = np.where(ahead, scene_focal * x / np.maximum(z, 1e-6) + scene.shape[1] / 2, -1).astype(np.float32)
    map_y = np.where(ahead, scene_focal * y / np.maximum(z, 1e-6) + scene.shape[0] / 2, -1).astype(np.float32)
    return cv2.remap(scene, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def as_a_camera_sees(picture: np.ndarray) -> np.ndarray:
    """A little blur and sensor noise: no real camera gives the clean edges cv2 draws."""
    blurred = cv2.GaussianBlur(picture, (3, 3), 0).astype(np.float64)
    return np.clip(blurred + np.random.default_rng(0).normal(0, 4, picture.shape), 0, 255).astype(np.uint8)


def test_camera() -> None:
    print("camera: find the game on the screen, and where Pupper stands")
    scene = [(9.0, [P.HURDLE, P.FREE, P.WALL]), (30.0, [P.BEAM, P.BEAM, P.BEAM])]
    view = draw_view(scene, width=480, height=600)
    width_m = 0.20
    focal = (S.PINHOLE_W / 2) / np.tan(np.radians(S.PINHOLE_HFOV_DEG / 2))
    K = np.array([[focal, 0, S.PINHOLE_W / 2], [0, focal, S.PINHOLE_H / 2], [0, 0, 1.0]])

    picture, truth = camera_picture(view, left=0.0, distance=0.5, facing_right_deg=0.0, width_m=width_m, focal=focal, size=(S.PINHOLE_W, S.PINHOLE_H))
    corners = S.find_frame(as_a_camera_sees(picture))
    error = None if corners is None else float(np.abs(corners - truth).max())
    check(f"the green frame's corners are found (it is {truth[1, 0] - truth[0, 0]:.0f} px wide here), within 1.5 px",
          error is not None and error < 1.5, error, "< 1.5 px")
    check("no frame in the picture: None, not a guess", S.find_frame(np.full((S.PINHOLE_H, S.PINHOLE_W, 3), 90, np.uint8)) is None, "a frame", None)

    worst, signs = 0.0, []
    for left, distance, facing in [(0.0, 0.5, 0.0), (0.25, 0.5, 0.0), (-0.25, 0.5, 0.0), (0.0, 0.7, 10.0), (0.2, 0.6, -8.0), (-0.25, 0.6, 12.0)]:
        picture, _ = camera_picture(view, left, distance, facing, width_m, focal, (S.PINHOLE_W, S.PINHOLE_H))
        picture = as_a_camera_sees(picture)
        position = S.locate(S.find_frame(picture), K, width_m)
        worst = max(worst, abs(position.left - left), abs(position.distance - distance), abs(position.facing_right_deg - facing) / 100)
        if left:
            signs.append(np.sign(position.left) == np.sign(left))
        if facing:
            signs.append(np.sign(position.facing_right_deg) == np.sign(facing))
    check("where the camera stands: within 3 cm and 3 degrees, in the middle and in both side lanes", worst < 0.03, f"worst {worst:.3f}", "< 0.03")
    check("signs: left of the centre line is +left, turned clockwise is +facing_right", all(signs), signs, "all True")

    # The whole chain, through the fisheye lens, from the centre and from a side lane.
    for left, label in [(0.0, "the centre lane"), (0.25, "the left lane")]:
        big_focal = 2 * focal
        scene_picture, _ = camera_picture(view, left, 0.5, 0.0, width_m, big_focal, (2 * S.PINHOLE_W, 2 * S.PINHOLE_H))
        fisheye = as_a_camera_sees(fisheye_picture(scene_picture, big_focal))
        sighting = S.ScreenFinder(view_width_m=width_m).find(fisheye)
        if sighting is None:
            check(f"through the fisheye, from {label}: the game is found", False, None, "a sighting")
            continue
        frame_px = float(np.linalg.norm(sighting.corners[1] - sighting.corners[0]))
        p = sighting.position
        check(f"through the fisheye, from {label}: position within 4 cm (the frame is only {frame_px:.0f} px wide)",
              abs(p.left - left) < 0.04 and abs(p.distance - 0.5) < 0.04, (round(p.left, 3), round(p.distance, 3)), (left, 0.5))
        hazards = P.detect(sighting.view)
        seen = (cells_seen(hazards, 9.0, 0.25), cells_seen(hazards, 30.0, 0.3))
        check("... and the game inside the frame is read correctly", seen == (scene[0][1], scene[1][1]),
              [(h.lane, h.kind, round(h.z, 1)) for h in hazards], scene)
        near = [h.z for h in hazards if h.z < 15]
        check("... with the near row's distance within 1.5 units of 9", bool(near) and max(abs(z - 9.0) for z in near) < 1.5,
              [round(z, 1) for z in near], "9.0 +- 1.5")


TESTS = {"perception": [test_perception, test_tracker], "policy": [test_policy], "moves": [test_moves],
         "squat": [test_squat], "camera": [test_camera]}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(TESTS)
    for name in wanted:
        if name not in TESTS:
            sys.exit(f"unknown part {name!r}; choose from {', '.join(TESTS)}")
        for test in TESTS[name]:
            test()
    print(f"\n{COUNT[0] - len(FAILED)} of {COUNT[0]} checks passed" + (f"; FAILED: {len(FAILED)}" if FAILED else ""))
    sys.exit(1 if FAILED else 0)
