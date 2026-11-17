"""Desk-side tests for the real game: no robot, no ROS, no browser, and no game.

    python3 tests/test_real.py

Like tests/test.py: every check prints what it wanted and what it got.

There are no pictures of the real game in this repository (they are not ours
to hand out). The pictures here are drawn by the test: flat track in the three
lanes where the reader expects them, a block of colour for a train, and the
three kinds of striped board. They check that the reader and the pilot do what
they say; whether that is good enough for the real game is another question,
answered only by playing it (real/README.md has the numbers).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "surfer"))

import real_learned as L  # noqa: E402
import real_perception as R  # noqa: E402
import real_policy  # noqa: E402
import server  # noqa: E402
from robot_io import SimRobot  # noqa: E402
from screen_finder import ScreenFinder  # noqa: E402

FAILED = []
COUNT = [0]

TRACK = (86, 91, 120)      # BGR: the grey-mauve of the track
GRASS = (40, 70, 30)
TRAIN = (200, 90, 40)
RED = (126, 96, 239)
WHITE = (250, 250, 250)


def check(name: str, ok: bool, got=None, want=None) -> None:
    COUNT[0] += 1
    if ok:
        print(f"  ok    {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}\n          got:    {got}\n          wanted: {want}")


# ----------------------------------------------------------------------- drawing
def picture(lane: int = 1, hud: bool = True) -> np.ndarray:
    """Open track in all three lanes, seen with the runner in `lane`."""
    view = np.full((R.VIEW_H, R.VIEW_W, 3), GRASS, np.uint8)
    for track in range(3):
        offset = track - lane
        top, bottom = R.SCAN_TOP_Y - 12, R.VIEW_H - 1
        corners = [(R.lane_x(offset, y, lane) + side * 0.5 * R.lane_pitch(y), y)
                   for y, side in ((top, -1), (top, 1), (bottom, 1), (bottom, -1))]
        cv2.fillPoly(view, [np.array(corners, np.int32)], TRACK)
    if hud:
        cv2.rectangle(view, (6, 6), (38, 38), (200, 150, 40), -1)     # the pause button
        cv2.rectangle(view, (570, 6), (636, 28), (20, 20, 20), -1)    # the score
    return view


def train(view: np.ndarray, offset: int, y: float, lane: int = 1) -> None:
    """Something tall standing in a lane, its foot at row y."""
    half = 0.5 * R.lane_pitch(y)
    x = R.lane_x(offset, y, lane)
    cv2.rectangle(view, (int(x - half), int(y - 2.0 * half)), (int(x + half), int(y)), TRAIN, -1)


def board_x(kind: str, offset: int, width: float, lane: int = 1) -> float:
    """The column a board of this width stands at, if it is in the lane `offset` over."""
    return R.RUNNER_X[lane] + offset * R.lane_pitch(R.FEET_Y) * width / R.WIDTH_AT_RUNNER[kind]


def board(view: np.ndarray, kind: str, offset: int, width: int, lane: int = 1, y: int = 190) -> None:
    x0 = int(board_x("jump" if kind == "low" else kind, offset, width, lane) - width / 2)
    if kind == "roll":      # tall, with two chevrons pointing down
        height = width
        cv2.rectangle(view, (x0, y), (x0 + width, y + height), RED, -1)
        for top in (0.15, 0.55):
            points = [(x0 + 2, y + int(top * height)), (x0 + width // 2, y + int((top + 0.3) * height)), (x0 + width - 2, y + int(top * height))]
            cv2.polylines(view, [np.array(points, np.int32)], False, WHITE, max(2, width // 9))
    elif kind == "bar":     # low and wide, upright stripes
        height = int(width / 2.6)
        cv2.rectangle(view, (x0, y), (x0 + width, y + height), WHITE, -1)
        stripe = width / 7
        for i in range(0, 7, 2):
            cv2.rectangle(view, (int(x0 + i * stripe), y), (int(x0 + (i + 1) * stripe), y + height), RED, -1)
    else:                   # low and wide, one chevron pointing up
        height = int(width / 2.6)
        cv2.rectangle(view, (x0, y), (x0 + width, y + height), RED, -1)
        points = [(x0 + 3, y + height - 2), (x0 + width // 2, y + 2), (x0 + width - 3, y + height - 2)]
        cv2.polylines(view, [np.array(points, np.int32)], False, WHITE, max(2, height // 3))


def scene(boards=(), blocked=None, playing=True) -> R.Scene:
    return R.Scene(boards=list(boards), blocked_y={-1: None, 0: None, 1: None, **(blocked or {})}, playing=playing)


def a_board(kind: str, seconds: float, lane: int = 0) -> R.Board:
    width = 1.0 / (seconds * R.CLOSING_RATE[kind] + 1.0 / R.WIDTH_AT_RUNNER[kind])
    return R.Board(kind, lane, 320.0, 180.0, width, width / 2.6, seconds)


def row_at(seconds: float) -> float:
    """The row at which something standing on the track is `seconds` away."""
    return min(range(R.SCAN_TOP_Y, R.FEET_Y + 1), key=lambda y: abs(R.seconds_from_row(y) - seconds))


# ----------------------------------------------------------------------- the reader
def test_reader() -> None:
    print("the reader: open track")
    s = R.read(picture())
    check("open track in all three lanes: nothing ahead", all(v is None for v in s.blocked_y.values()) and not s.boards, s.summary(), "open, no boards")
    check("the pause button and the score are there: a run is on", s.playing and R.screen(picture()) == "run", R.screen(picture()), "run")
    check("without them it is not a run (a menu)", not R.read(picture(hud=False)).playing and R.screen(picture(hud=False)) == "menu", R.screen(picture(hud=False)), "menu")
    score = picture(hud=False)
    x0, y0, x1, y1 = R.PLAY_BUTTON
    cv2.rectangle(score, (x0, y0), (x1, y1), (60, 200, 60), -1)
    check("a green PLAY button where the score screen has one: the score screen", R.screen(score) == "score", R.screen(score), "score")

    view = picture()
    train(view, 0, 200)
    s = R.read(view)
    check("a train in my lane, its foot at row 200: my lane ends there, the others are open",
          s.blocked_y[0] is not None and abs(s.blocked_y[0] - 200) <= 5 and s.blocked_y[-1] is None and s.blocked_y[1] is None, s.summary(), "+0:200, others open")
    check("... and that is seconds_from_row(200) away", s.seconds(0) is not None and abs(s.seconds(0) - R.seconds_from_row(200)) < 0.06, s.seconds(0), R.seconds_from_row(200))
    view = picture()
    train(view, -1, 300)
    s = R.read(view)
    check("a train beside me on the left: the left lane is closed right here", s.blocked_y[-1] is not None and s.blocked_y[-1] > 280 and s.blocked_y[0] is None, s.summary(), "-1 closed near row 300")
    clock = [R.seconds_from_row(y) for y in range(R.SCAN_TOP_Y, R.FEET_Y + 1, 10)]
    check("further up the picture is further away in time, and the runner's feet are 0 s away",
          all(a > b for a, b in zip(clock, clock[1:])) and R.seconds_from_row(R.FEET_Y) < 0.03 and 0.8 < clock[0] < 2.5, [round(c, 2) for c in clock], "falling to 0")

    for lane, name in ((0, "left"), (2, "right")):
        s = R.read(picture(lane), lane)
        off = -1 if lane == 0 else 1
        check(f"runner in the {name} lane: my lane and the centre are open, and there is no track beyond the edge",
              s.blocked_y[0] is None and s.blocked_y[-off] is None and s.blocked_y[off] is not None and s.blocked_y[off] > 300, s.summary(), "edge side closed")

    reader = R.Reader()
    reader.read(picture())
    hidden = picture()
    cv2.circle(hidden, (R.RUNNER_X[1], 311), 60, (30, 200, 240), -1)   # a big coin over the runner and the track beside it
    s = reader.read(hidden)
    check("the track beside the runner hidden for a moment: the colour from before is used", s.track_ok and s.blocked_y[-1] is None and s.blocked_y[1] is None, s.summary(), "side lanes still open")
    roof = np.full_like(picture(), (60, 40, 150))
    cv2.rectangle(roof, (6, 6), (38, 38), (200, 150, 40), -1)
    for _ in range(R.TRACK_NEW_AFTER):
        s = reader.read(roof)
    check("a new colour that stays (running on a train's roof) is believed after a moment", s.blocked_y[0] is None, s.summary(), "+0:open")

    print("the reader: striped boards")
    for kind, width, want in (("jump", 40, "jump"), ("roll", 34, "roll"), ("bar", 56, "bar"), ("low", 20, "low")):
        view = picture()
        board(view, kind, 0, width)
        found = R.read(view).boards
        b = found[0] if found else None
        check(f"a {kind} board {width} px wide in my lane is read as '{want}', {width} px wide",
              len(found) == 1 and b.kind == want and b.lane == 0 and abs(b.w - width) <= 3, [(f.kind, f.lane, f.w) for f in found], (want, 0, width))
    for offset in (-1, 1):
        view = picture()
        board(view, "jump", offset, 36)
        found = R.read(view).boards
        check(f"a board in the lane {'left' if offset < 0 else 'right'} of mine is put in that lane", len(found) == 1 and found[0].lane == offset, [(f.kind, f.lane) for f in found], offset)
    widths = [R.seconds_from_width(w, "jump") for w in (20, 30, 40, 50)]
    check("a wider board is a nearer board", all(a > b for a, b in zip(widths, widths[1:])) and R.seconds_from_width(R.WIDTH_AT_RUNNER["jump"], "jump") == 0, widths, "falling to 0")
    view = picture()
    cv2.rectangle(view, (0, 120), (140, 260), RED, -1)   # a red train side, or a red sky
    check("a big red thing that is not a board is not a board", not R.read(view).boards, [(f.kind, f.w) for f in R.read(view).boards], [])


# ----------------------------------------------------------------------- the pilot
def fresh_pilot(lane: int = 1) -> real_policy.Pilot:
    pilot = real_policy.Pilot()
    pilot.step(scene(), 0.0)   # the run starts
    pilot.lane = lane
    return pilot


def test_pilot() -> None:
    print("the pilot")
    lead = real_policy.JUMP_LEAD_S
    pilot = real_policy.Pilot()
    first = pilot.step(scene([a_board("jump", 0.1)]), 0.0)
    check("the first second of a run is not acted on (a flash, the guard in the picture)", first is None and pilot.step(scene([a_board("jump", 0.1)]), 0.5) is None, first, None)
    check("after it, it is", pilot.step(scene([a_board("jump", 0.1)]), 1.1) == "jump", "no jump", "jump")

    pilot = fresh_pilot()
    early = pilot.step(scene([a_board("jump", lead + 0.3)]), 2.0)
    due = pilot.step(scene([a_board("jump", lead - 0.02)]), 2.3)
    check("a jump board: wait, then jump when it is JUMP_LEAD_S away", early is None and due == "jump", (early, due), (None, "jump"))
    check("while the runner is in the air the picture is not acted on", pilot.step(scene([a_board("roll", 0.1)]), 2.5) is None, "a move", None)
    check("a roll board: squat", fresh_pilot().step(scene([a_board("roll", 0.2)]), 2.0) == "squat", "not squat", "squat")
    check("a board too far to tell (low) that gets close anyway: jump", fresh_pilot().step(scene([a_board("low", 0.2)]), 2.0) == "jump", "not jump", "jump")
    check("a board in another lane is not mine to jump", fresh_pilot().step(scene([a_board("jump", 0.1, lane=1)]), 2.0) is None, "a move", None)

    wall, beside = row_at(0.6), R.FEET_Y
    pilot = fresh_pilot()
    move = pilot.step(scene(blocked={0: wall, -1: beside}), 2.0)
    check("a train ahead, a train beside me on the left, the right lane open: step right", move == "right" and pilot.lane == 2, (move, pilot.lane), ("right", 2))
    pilot = fresh_pilot()
    move = pilot.step(scene(blocked={0: wall, 1: row_at(0.9)}), 2.0)
    check("a train ahead, the left lane open and the right open for less long: step left", move == "left" and pilot.lane == 0, (move, pilot.lane), ("left", 0))
    check("a train still far away: keep running", fresh_pilot().step(scene(blocked={0: row_at(real_policy.DODGE_LEAD_S + 0.4)}), 2.0) is None, "a move", None)
    pilot = fresh_pilot(lane=2)
    move = pilot.step(scene(blocked={0: wall, -1: beside}), 2.0)
    check("in the right lane with a train ahead and one on the left: there is no lane further right", move is None and pilot.lane == 2, (move, pilot.lane), (None, 2))
    pilot = fresh_pilot()
    move = pilot.step(scene([a_board("bar", 0.5)], blocked={0: wall}), 2.0)
    check("a striped bar is the end of a track: go around, don't jump", move in ("left", "right"), move, "left or right")
    pilot = fresh_pilot()
    move = pilot.step(scene([a_board("jump", 0.5, lane=1)], blocked={0: wall, -1: row_at(0.55), 1: row_at(0.5)}), 2.0)
    check("a train ahead; left a train a little further, right a board to jump: the board's lane", move == "right", move, "right")

    pilot = fresh_pilot()
    check("not in a run (menu, score screen): no moves", pilot.step(scene([a_board("jump", 0.1)], playing=False), 2.0) is None, "a move", None)
    check("... and the next run gets its first second again", pilot.step(scene([a_board("jump", 0.1)]), 2.1) is None, "a move", None)

    # The game twice as fast: a board grows twice as fast, and has half the time its width says.
    pilot = fresh_pilot()
    for i in range(12):
        t = 2.0 + i / 30
        pilot.step(scene([a_board("jump", 1.6 - 2.0 * (t - 2.0))]), t)
    check("boards growing twice as fast: the game's speed is measured as about 2", 1.5 < pilot.speed < 2.3, pilot.speed, "about 2")
    slow, fast = fresh_pilot(), fresh_pilot()
    fast.speed = 2.0
    b = a_board("jump", 2 * lead - 0.05)
    check("at twice the speed the same picture means half the time: jump sooner", slow.step(scene([b]), 2.0) is None and fast.step(scene([b]), 2.0) == "jump", "same", "fast jumps, slow waits")


# ----------------------------------------------------------------------- the dog behind the game
def test_body() -> None:
    print("the dog follows the game")
    sent = []

    class Page:
        async def send_text(self, text: str) -> None:
            sent.append(json.loads(text))

        async def send_bytes(self, _data: bytes) -> None:
            pass

    async def play() -> dict:
        robot = SimRobot()
        player = server.Player(robot, "stream", False, ScreenFinder(), game="real")
        player.page = Page()
        out = {}
        for moves, want in ((["left", "right"], []), (["left", "jump", "right"], ["left", "jump", "right"]),
                            (["jump", "squat", "jump", "left"], ["squat", "jump", "left"]), (["left", "left", "right", "squat"], ["left", "squat"])):
            player.body_queue.clear()
            for move in moves:
                player.queue_body(move)
            out[" ".join(moves)] = (list(player.body_queue), want)
        player.body_queue.clear()

        await robot.stand_up()
        await player.start(None)
        open_track = picture()
        now_board = picture()
        board(now_board, "jump", 0, 44)
        await player.see_real(open_track)
        await asyncio.sleep(real_policy.START_GRACE_S + 0.1)
        await player.see_real(now_board)          # the key must go out on this very picture
        await asyncio.sleep(0.05)
        out["key"] = [m for m in sent if m.get("type") == "move"]
        out["queue"] = list(player.body_queue)
        act = asyncio.create_task(player.act())
        player.last_page_at = 1e18                # the page is "still there"
        await asyncio.sleep(1.2)
        out["played"] = [name for _t, name in robot.references]
        out["body"] = [m for m in sent if m.get("type") == "body"]
        out["status"] = player.status()
        await player.see_real(picture(hud=False))
        await asyncio.sleep(server.RUN_OVER_S + 0.2)
        await player.see_real(picture(hud=False))
        out["after"] = player.status()
        act.cancel()
        return out

    out = asyncio.run(play())
    for moves in ("left right", "left jump right", "jump squat jump left", "left left right squat"):
        got, want = out[moves]
        check(f"the dog owes [{moves}] -> it will make {want}", got == want, got, want)
    check("a jump board close ahead: the key goes to the game on that picture", len(out["key"]) == 1 and out["key"][0]["move"] == "jump", out["key"], "one jump")
    check("... and the dog starts the same jump within a moment", len(out["body"]) == 1 and out["body"][0]["move"] == "jump" and out["queue"] == ["jump"], (out["body"], out["queue"]), "a jump")
    check("... by playing the jump table on the robot", out["played"] == ["jump"], out["played"], ["jump"])
    check("the status says a run is on", out["status"]["run"] is True and out["status"]["game"] == "real", out["status"], "run: True")
    check("no run on screen for a while: the run is over", out["after"]["run"] is False, out["after"]["run"], False)


# ----------------------------------------------------------------------- slow motion
def test_clock() -> None:
    print("slow motion: the server counts game seconds")

    class Page:
        async def send_text(self, _text: str) -> None:
            pass

        async def send_bytes(self, _data: bytes) -> None:
            pass

    async def play() -> dict:
        player = server.Player(SimRobot(), "stream", False, ScreenFinder(), game="real")
        player.page = Page()
        out = {"before": player.status()["clock"]}
        await player.on_message({"type": "state", "game": "real", "clock": 0.5})
        out["after"] = player.status()["clock"]
        await player.see_real(picture())
        start = player.game_now
        await asyncio.sleep(0.4)
        await player.see_real(picture())
        out["half"] = player.game_now - start
        await player.on_message({"type": "state", "game": "real", "clock": 1.0})
        await player.see_real(picture())
        start = player.game_now
        await asyncio.sleep(0.4)
        await player.see_real(picture())
        out["full"] = player.game_now - start
        return out

    out = asyncio.run(play())
    check("the game's speed is 1 until real/play.py says otherwise", out["before"] == 1.0, out["before"], 1.0)
    check("play.py says the game runs at half speed: the status says so too", out["after"] == 0.5, out["after"], 0.5)
    check("0.4 s at half speed is about 0.2 game seconds", 0.17 < out["half"] < 0.26, out["half"], "about 0.2")
    check("0.4 s at full speed is about 0.4 game seconds", 0.37 < out["full"] < 0.50, out["full"], "about 0.4")


# ----------------------------------------------------------------------- learned eyes
def model_file(folder: Path, favourite: str | None, seed: int = 0) -> Path:
    """A model file of the right shape. With a favourite, it always answers that move
    (all weights zero, the last layer's bias set); without, random numbers."""
    rng = np.random.default_rng(seed)
    arrays, channels, height, width, size = [], 6, L.INPUT_H, L.INPUT_W, None
    for layer in L.LAYERS:
        if layer[0] == "conv":
            _, outputs, kernel, stride = layer
            arrays.append((rng.normal(0, 0.1, (outputs, channels, kernel, kernel)), rng.normal(0, 0.1, outputs)))
            channels, height, width = outputs, -(-height // stride), -(-width // stride)
        else:
            size = size or channels * height * width
            arrays.append((rng.normal(0, 0.05, (layer[1], size)), rng.normal(0, 0.1, layer[1])))
            size = layer[1]
    if favourite:
        arrays = [(np.zeros_like(w), np.zeros_like(b)) for w, b in arrays]
        arrays[-1][1][L.MOVES.index(favourite)] = 5.0
    path = folder / f"model-{favourite or 'random'}.npz"
    L.save(path, arrays, {"threshold": 0.6})
    return path


def test_learned(folder: Path) -> None:
    print("learned eyes: the network's arithmetic and how it is used")
    rng = np.random.default_rng(1)
    x = rng.random((3, 7, 9)).astype(np.float32)
    w = rng.normal(size=(2, 3, 3, 3)).astype(np.float32)
    b = np.array([0.5, -0.5], np.float32)
    fast = L.convolve(x, w, b, 2)
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    slow = np.array([[[(w[o] * padded[:, 2 * i:2 * i + 3, 2 * j:2 * j + 3]).sum() + b[o] for j in range(5)] for i in range(4)] for o in range(2)])
    check("a convolution gives what adding it up by hand gives", fast.shape == (2, 4, 5) and np.allclose(fast, slow, atol=1e-4), fast.shape, "(2, 4, 5), same numbers")

    net = L.Net(model_file(folder, None))
    answer = net(L.pair(L.small(picture()), L.small(picture())))
    check("the network answers with five likelihoods that add up to 1", answer.shape == (5,) and abs(answer.sum() - 1) < 1e-5 and (answer >= 0).all(), answer, "5 numbers, sum 1")
    began = time.perf_counter()
    for _ in range(5):
        net(L.pair(L.small(picture()), L.small(picture())))
    each = (time.perf_counter() - began) / 5
    check("one picture takes well under a tenth of a second on a laptop", each < 0.1, f"{each * 1000:.0f} ms", "< 100 ms")

    check("nothing likely enough: no press", L.choose(np.array([0.5, 0.2, 0.1, 0.1, 0.1]), 1.0, 0.0, 0.6) is None, "a press", None)
    check("a move likely enough: press it", L.choose(np.array([0.1, 0.1, 0.1, 0.65, 0.05]), 1.0, 0.0, 0.6) == "jump", "", "jump")
    check("just after a press: no press, whatever the picture says", L.choose(np.array([0.1, 0.1, 0.1, 0.65, 0.05]), 1.0, 1.2, 0.6) is None, "a press", None)

    pilot = L.LearnedPilot(model_file(folder, "left"))
    view = picture()
    moves = [pilot.step(view, t) for t in (0.0, 0.1, 0.2, L.REFRACTORY_S + 0.01)]
    check("a model that always says left: left, then nothing while the runner is mid-move, then left again",
          moves == ["left", None, None, "left"], moves, ["left", None, None, "left"])
    check("... and it keeps count of the lane (it cannot go further left than the left lane)", pilot.lane == 0, pilot.lane, 0)
    check("the threshold comes from the model file", pilot.threshold == 0.6, pilot.threshold, 0.6)

    # The older picture of the pair: the newest one that is at least GAP_S old.
    pilot = L.LearnedPilot(model_file(folder, "none"))
    seen = []
    pilot.net = lambda pair: seen.append(pair) or np.array([1.0, 0, 0, 0, 0])
    for index, at in enumerate((0.00, 0.04, 0.08, 0.12, 0.16)):
        pilot.step(np.full((L.INPUT_H, L.INPUT_W, 3), index * 40, np.uint8), at)
    older = [round(float(pair[3, 0, 0]) * 255 / 40) for pair in seen]
    check("the second picture of each pair is the newest one at least a tenth of a game second old",
          older == [0, 0, 0, 0, 1], older, [0, 0, 0, 0, 1])

    sent = []

    class Page:
        async def send_text(self, text: str) -> None:
            sent.append(json.loads(text))

        async def send_bytes(self, _data: bytes) -> None:
            pass

    async def play() -> dict:
        robot = SimRobot()
        player = server.Player(robot, "stream", False, ScreenFinder(), game="real", model=str(model_file(folder, "squat")))
        player.page = Page()
        await robot.stand_up()
        await player.start(None)
        await player.see_real(picture(hud=False))     # no run on screen: no press
        nothing = [m for m in sent if m.get("type") == "move"]
        await player.see_real(picture())
        await asyncio.sleep(0.05)
        return {"nothing": nothing, "moves": [m for m in sent if m.get("type") == "move"], "queue": list(player.body_queue),
                "status": player.status(), "debug": player.debug}

    out = asyncio.run(play())
    check("with a model, no run on screen: no key", out["nothing"] == [], out["nothing"], [])
    check("with a model that says squat: the key goes to the game and the dog owes a squat",
          [m["move"] for m in out["moves"]] == ["squat"] and out["queue"] == ["squat"], (out["moves"], out["queue"]), "squat")
    check("the status says which eyes are in use", out["status"]["eyes"] == "learned", out["status"]["eyes"], "learned")
    check("the picture for the strip is drawn", out["debug"] is not None and out["debug"].shape == (360, 640, 3), None, "640 x 360")


# ----------------------------------------------------------------------- recording and training
def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "real" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def lesson(lane_with_board: int | None) -> np.ndarray:
    """A drawn picture for the training check: open track, or a jump board close ahead in one lane."""
    view = picture()
    if lane_with_board is not None:
        board(view, "jump", lane_with_board - 1, 44)
    return view


def test_recording(folder: Path) -> None:
    print("recording a person's play")
    try:
        play = load_module("play")
    except ImportError as error:   # real/play.py needs playwright; the robot does not have it and does not need it
        print(f"  skipped: real/play.py can't be loaded here ({error})")
        return
    recorder = play.Recorder(folder / "recordings")
    recorder.add(picture(), True, 0.5, 100.0)
    check("recording off: nothing is saved", not (folder / "recordings").exists(), "a folder", "nothing")

    # A run in slow motion (clock 0.5): 80 pictures over 8 real seconds = 4 game seconds, a key at real 3.0 s.
    recorder.on = True
    for index in range(81):
        real = 200.0 + index * 0.1
        recorder.add(picture(), True, 0.5, real)
        if index == 30:
            recorder.key("jump", real + 0.02)
    run = recorder.run
    for index in range(12):                               # the crash: no run on screen
        recorder.add(picture(hud=False), False, 1.0, 208.1 + index * 0.1)
    info = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    check("a run is saved when it ends", bool(info) and recorder.run is None, info.keys(), "run.json")
    check("times are GAME seconds: 8 real seconds at half speed is 4", abs(info.get("seconds", 0) - 4.0) < 0.06, info.get("seconds"), 4.0)
    check("one picture a tick, each with its time", len(info.get("times", [])) == 81 and len(list(run.glob("*.jpg"))) == 81, len(info.get("times", [])), 81)
    check("the key is saved at its game time (real 3.02 s -> 1.51 game s)", len(info.get("keys", [])) == 1 and info["keys"][0][1] == "jump"
          and abs(info["keys"][0][0] - 1.51) < 0.02, info.get("keys"), [[1.51, "jump"]])
    saved = cv2.imread(str(run / "00000.jpg"))
    check("pictures are saved small", saved is not None and (saved.shape[1], saved.shape[0]) == play.RECORD_SIZE, saved is not None and saved.shape, play.RECORD_SIZE)

    for index in range(5):                                # a run of half a game second: too short to teach anything
        recorder.add(picture(), True, 1.0, 300.0 + index * 0.1)
    short = recorder.run
    for index in range(12):
        recorder.add(picture(hud=False), False, 1.0, 300.5 + index * 0.1)
    check("a run that lasted under two game seconds is thrown away", not short.exists(), "kept", "gone")
    check("the summary counts what was kept", recorder.summary().startswith("recording (waiting for a run); 1 run saved"), recorder.summary(), "1 run saved")

    print("from recordings to a model")
    train_py = load_module("train")
    times = np.round(np.arange(0, 6.0, 0.025), 4)
    labels, counted = train_py.label(times, [[2.0, "left"], [5.6, "jump"]], 6.0)
    at = lambda t: int(np.argmin(np.abs(times - t)))  # noqa: E731
    check("the pictures just before a key press are examples of that key", labels[at(1.95)] == L.MOVES.index("left") and counted[at(1.95)], labels[at(1.95)], "left")
    check("pictures a little further from it are not counted either way", not counted[at(1.85)] and not counted[at(2.1)], (counted[at(1.85)], counted[at(2.1)]), "not counted")
    check("pictures far from any key are examples of pressing nothing", labels[at(1.0)] == 0 and counted[at(1.0)], labels[at(1.0)], "none")
    check("the last second of a run (the crash) teaches nothing", not counted[at(5.5)] and not counted[at(5.59)], counted[at(5.5)], False)
    probabilities = np.tile([0.9, 0.025, 0.025, 0.025, 0.025], (len(times), 1))
    probabilities[at(1.9):at(2.3)] = [0.1, 0.8, 0.03, 0.04, 0.03]
    probabilities[at(3.0):at(3.05)] = [0.1, 0.03, 0.03, 0.8, 0.04]
    pressed = train_py.presses(times, probabilities, 0.6)
    check("a move that stays likely for a while is pressed once, then again only after the runner has moved",
          [move for _t, move in pressed] == ["left", "left", "jump"], pressed, "left, left, jump")
    check("scoring: the person's left was made, and the second left and the jump are extra",
          train_py.compare([[2.0, "left"]], pressed, 6.0) == (1, 0, 2), train_py.compare([[2.0, "left"]], pressed, 6.0), (1, 0, 2))

    if train_py.torch is None:
        print("  skipped: no PyTorch here, so the training itself is not run (pip install -r real/requirements-train.txt)")
        return
    # Six drawn runs: every 1.5 game s a jump board appears close ahead in the runner's lane, and "the person" jumps.
    lessons = folder / "lessons"
    for number in range(6):
        run = lessons / f"run{number:02d}"
        run.mkdir(parents=True)
        times, keys = [], []
        for index in range(240):
            at_s = index * 0.025
            tick = (index + 8 * number) % 60                # the pattern repeats every 60 pictures (1.5 game s)
            shown = 36 <= tick < 42
            cv2.imwrite(str(run / f"{index:05d}.jpg"), cv2.resize(lesson(1 if shown else None), play.RECORD_SIZE))
            times.append(round(at_s, 4))
            if tick == 41:
                keys.append([round(at_s, 4), "jump"])       # pressed as the board's last picture shows
        (run / "run.json").write_text(json.dumps({"size": list(play.RECORD_SIZE), "seconds": times[-1], "clock": 1.0, "times": times, "keys": keys}))
    result = train_py.teach([lessons], folder / "lessons.npz", epochs=6)
    asked = result["made"] + result["missed"]
    check("training on drawn lessons: the model makes the kept-aside run's jumps", asked >= 2 and result["made"] == asked, result, "all made")
    check("... with no extra presses", result["extra"] == 0, result["extra"], 0)
    check("the saved file gives the same answers through numpy as through PyTorch", result["agreement"] < 1e-4, result["agreement"], "< 1e-4")
    pilot = L.LearnedPilot(folder / "lessons.npz")
    quiet = [pilot.step(lesson(None), 0.025 * index) for index in range(8)]
    pilot.reset()
    loud = [pilot.step(lesson(1), 0.025 * index) for index in range(8)]
    check("played back: open track, no press; a board close ahead, a jump", not any(quiet) and "jump" in loud, (quiet, loud), "nothing; jump")


if __name__ == "__main__":
    test_reader()
    test_pilot()
    test_body()
    test_clock()
    scratch = Path(tempfile.mkdtemp(prefix="pupper-real-"))
    try:
        test_learned(scratch)
        test_recording(scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"\n{COUNT[0] - len(FAILED)} of {COUNT[0]} checks passed" + (f"; FAILED: {len(FAILED)}" if FAILED else ""))
    sys.exit(1 if FAILED else 0)
