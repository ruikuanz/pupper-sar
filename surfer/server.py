"""Pupper plays Lane Hound: the server that runs on the robot (or, with --sim, on a laptop).

    python3 surfer/server.py --sim                 # laptop: a pretend robot, eyes on the streamed game
    python3 surfer/server.py                       # Pupper: real moves, eyes on the streamed game
    python3 surfer/server.py --source camera       # Pupper: real moves, eyes on its own camera
    python3 surfer/server.py --game real           # Pupper plays the real Subway Surfers (see real/README.md)
    python3 surfer/server.py --game real --model real/model.npz    # ... deciding with what it learned from a person

Then open the game: http://localhost:8770 (from a laptop, through
`ssh -N -L 8770:localhost:8770 pi@pupper<N>.local`).

It does three things at once (three asyncio tasks):

    see     a picture -> the obstacle rows ahead           (screen_finder.py, perception.py)
    act     20 times a second: pick a move, command the legs   (policy.py, moves.py)
    report  tell the game page what Pupper is doing, and show what it sees

Where the picture comes from (--source):

    stream  the game page sends Pupper's view of the game over the WebSocket.
            Nothing to line up; works anywhere.
    camera  Pupper looks at the screen with its own camera, finds the green
            frame, and reads the game inside it. It also learns where it is
            standing, and with --hold it steers back to its lane.

Safety: Pupper only moves between "start" and "stop". It stops by itself if
the game page goes quiet, if it falls, or if the e-stop is pressed; the
joystick always outranks it. `curl -X POST localhost:8770/stop` stops it too.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

import perception
import real_learned
import real_perception
import real_policy
from moves import Position, State, Surfer
from policy import decide
from robot_io import RealRobot, SimRobot
from screen_finder import ScreenFinder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("surfer")

GAME_DIR = Path(__file__).resolve().parent.parent / "game"
ACT_RATE = 20.0        # Hz: the speed commands (cmd_vel_mux drops a source quiet for 0.5 s)
PAGE_TIMEOUT_S = 1.5   # the game page says something 4 times a second; quiet this long: stop
STALE_S = 0.4          # a picture older than this is not acted on
REPORT_S = 0.2         # how often the page hears from us
POSITION_BLEND = 0.4   # how much a new position measurement counts (the rest is the old one)
# --- the real game (--game real) ---------------------------------------------------
BODY_QUEUE_MAX = 3     # moves the dog still has to make to catch up with the runner on screen
RUN_OVER_S = 1.5       # no run on screen for this long: the run is over


class Player:
    """Pupper as a player: its eyes, its decisions, its legs."""

    def __init__(self, robot, source: str, hold: bool, finder: ScreenFinder, game: str = "lane-hound",
                 model: str | None = None) -> None:
        self.robot = robot
        self.source = source
        self.finder = finder
        self.game = game               # "lane-hound" (our own game) or "real" (Subway Surfers itself)
        self.surfer = Surfer(robot, self.emit, hold=hold)
        self.tracker = perception.RowTracker()
        # The real game: its own eyes and its own decisions (real_perception.py, real_policy.py).
        # With a model (real/train.py makes one from a person's play) the decisions come from
        # real_learned.py instead; real_perception.py then only says whether a run is on screen.
        self.reader = real_perception.Reader()
        self.learned = real_learned.LearnedPilot(model) if model else None
        self.pilot = self.learned or real_policy.Pilot()
        # real/play.py can slow the game down (--speed). It says how fast the game's clock is
        # running, and everything that is timed against the game uses game seconds.
        self.clock = 1.0                  # game seconds per real second
        self.game_now = 0.0               # game seconds since the server started
        self.clock_read_at: float | None = None
        self.body_queue: list[str] = []   # moves made in the game that the dog has yet to make
        self.run_on = False               # is a run on screen right now?
        self.last_run_at = 0.0
        self.page: WebSocket | None = None
        self.playing = False
        self.epoch = 0                 # counts stops: a start that was overtaken by a stop gives up
        self.message = ""              # why it last stopped
        self.last_page_at = 0.0
        self.stream_jpeg: bytes | None = None
        self.new_stream = asyncio.Event()
        self.seen_at = 0.0             # when the last picture was read
        self.screen_found: bool | None = None  # camera source: is the green frame in view?
        self.position: Position | None = None
        self.hazards: list = []
        self.debug: np.ndarray | None = None
        self.frames = 0
        self.fps = 0.0

    # ------------------------------------------------------------ to the page
    def emit(self, message: dict) -> None:
        """A move just started: tell the game (called by Surfer, which can't await)."""
        log.info("move %s (lane %d)", message["move"], message["lane"])
        if self.game == "real":
            # The real game got its key when the move was decided (see see_real):
            # this is the dog's body catching up, for the status line only.
            message = {**message, "type": "body"}
        asyncio.get_running_loop().create_task(self.send(message))

    async def send(self, message: dict | bytes) -> None:
        page = self.page
        if page is None:
            return
        try:
            if isinstance(message, bytes):
                await page.send_bytes(message)
            else:
                await page.send_text(json.dumps(message))
        except (WebSocketDisconnect, RuntimeError):
            pass  # the page went away; its socket handler cleans up

    def status(self) -> dict:
        problem = self.robot.problem()
        position = self.position
        real = {"game": "real", "run": self.run_on, "game_lane": self.pilot.lane, "behind": len(self.body_queue),
                "game_speed": round(self.pilot.speed, 2), "clock": round(self.clock, 2),
                "eyes": "learned" if self.learned else "rules"} if self.game == "real" else {"game": "lane-hound"}
        return {
            **real,
            "type": "status",
            "mode": self.robot.mode,
            "state": "playing" if self.playing else (problem or "standing, ready"),
            "ready": not problem,
            "move": self.surfer.state.name.lower(),
            "lane": self.surfer.lane,
            "source": self.source,
            "screen": self.screen_found,
            "fps": round(self.fps, 1),
            "speed": round(self.tracker.speed, 2),
            "rows": [[round(r.z, 1), r.cells] for r in self.tracker.rows],
            "position": position and {"left": round(position.left, 3), "distance": round(position.distance, 3),
                                      "facing_right_deg": round(position.facing_right_deg, 1)},
            "message": self.message,
        }

    # ------------------------------------------------------------ start, stop
    async def start(self, speed: float | None) -> None:
        """The page pressed start: stand up if need be, then play."""
        epoch = self.epoch
        ok, message = True, "ready"
        if self.robot.problem():
            ok, message = await self.robot.stand_up()
        if epoch != self.epoch:
            ok, message = False, "stopped while standing up"
        if ok:
            self.tracker.reset(speed)
            self.position = None
            self.pilot.reset()
            self.body_queue.clear()
            self.run_on = False
            self.surfer.start(time.monotonic())
            self.playing, self.message = True, ""
            log.info("playing (track speed %.1f)", self.tracker.nominal_speed)
        else:
            log.warning("can't start: %s", message)
        await self.send({"type": "ready", "ok": ok, "message": message})

    def stop(self, why: str) -> None:
        self.epoch += 1
        if self.playing:
            log.info("stopped: %s", why)
            self.message = why
        self.playing = False
        self.surfer.stop()

    async def on_message(self, message: dict) -> None:
        if message.get("type") == "state" and "clock" in message:
            self.clock = min(4.0, max(0.0, float(message["clock"])))   # how fast real/play.py is running the game
        if message.get("type") != "control":
            return  # other "state" messages only need to arrive (they keep last_page_at fresh)
        action = message.get("action")
        if action == "start":
            self.stop("restarting")
            asyncio.get_running_loop().create_task(self.start(message.get("speed")))
        elif action == "stop":
            self.stop("stopped from the game page")
        elif action == "stand_up":
            asyncio.get_running_loop().create_task(self.robot.stand_up())
        elif action == "lie_down":
            self.stop("lying down")
            asyncio.get_running_loop().create_task(self.robot.lie_down())

    # ------------------------------------------------------------ see
    def read(self, picture: np.ndarray, from_camera: bool):
        """One picture -> (hazards, view, position, debug picture). Runs in a worker thread."""
        position, corners, wide = None, None, None
        if from_camera:
            sighting = self.finder.find(picture)
            wide = self.finder.picture
            if sighting is None:
                return None, None, None, cv2.resize(wide, (perception.VIEW_W, perception.VIEW_W * 3 // 4))
            view, position, corners = sighting.view, sighting.position, sighting.corners
        else:
            view = cv2.resize(picture, (perception.VIEW_W, perception.VIEW_H), interpolation=cv2.INTER_AREA)
        hazards = perception.detect(view)
        return hazards, view, position, (wide, corners)

    # ------------------------------------------------------------ see (the real game)
    def queue_body(self, move: str) -> None:
        """The game has had its key; the dog owes the same move."""
        queue = self.body_queue
        opposite = {"left": "right", "right": "left"}.get(move)
        if opposite and queue and queue[-1] == opposite:
            queue.pop()   # left then right: the dog can stay where it is
            return
        queue.append(move)
        while len(queue) > BODY_QUEUE_MAX:
            tricks = [i for i, m in enumerate(queue) if m in ("jump", "squat")]
            if not tricks:
                break     # only lane changes left: the dog must make them all to end up in the right lane
            del queue[tricks[0]]

    async def see_real(self, picture: np.ndarray) -> None:
        """One picture of the real game: read it, and if a move is due, make it in the game at once.

        The game's runner moves in a tenth of a second; the dog needs most of a
        second. So the key goes to the game now and the dog follows as fast as
        it can (body_queue): it makes every move, a little behind."""
        if self.learned:
            scene = None
            running = await asyncio.to_thread(real_perception.screen, picture) == "run"
        else:
            scene = await asyncio.to_thread(self.reader.read, picture, self.pilot.lane)
            running = scene.playing
        now = time.monotonic()
        if self.clock_read_at is not None:
            self.game_now += (now - self.clock_read_at) * self.clock
        self.clock_read_at = now
        self.seen_at = now
        if running:
            if not self.run_on:
                log.info("a run has started")
                self.pilot.reset()
            self.run_on, self.last_run_at = True, now
        elif self.run_on and now - self.last_run_at > RUN_OVER_S:
            log.info("the run is over")
            self.run_on = False
            # Back to the middle for the next run.
            self.body_queue = ["left" if self.surfer.lane > 1 else "right"] * abs(self.surfer.lane - 1)
        if self.playing and self.run_on:
            if self.learned:
                move = await asyncio.to_thread(self.learned.step, picture, self.game_now)
            else:
                move = self.pilot.step(scene, self.game_now)
            if move:
                await self.send({"type": "move", "move": move, "lane": self.pilot.lane})
                self.queue_body(move)
        note = f"lane {self.pilot.lane}  dog {self.surfer.state.name.lower()}  behind {len(self.body_queue)}  game x{self.clock:.2f}"
        if self.learned:
            self.debug = real_learned.draw_debug(picture, self.learned.likely, self.learned.threshold, note)
        else:
            self.debug = real_perception.draw_debug(picture, scene, note)

    async def see(self) -> None:
        last_count_at = time.monotonic()
        while True:
            if self.source == "camera":
                picture = self.robot.frame()
                if picture is None:
                    await asyncio.sleep(0.01)
                    continue
            else:
                await self.new_stream.wait()
                self.new_stream.clear()
                picture = cv2.imdecode(np.frombuffer(self.stream_jpeg, np.uint8), cv2.IMREAD_COLOR)
                if picture is None:
                    continue
            if self.game == "real":
                await self.see_real(picture)
                self.frames += 1
                now = time.monotonic()
                if now - last_count_at >= 1.0:
                    self.fps, self.frames, last_count_at = self.frames / (now - last_count_at), 0, now
                continue
            hazards, view, position, extra = await asyncio.to_thread(self.read, picture, self.source == "camera")
            now = time.monotonic()
            if hazards is None:  # camera: the frame is not in view
                self.screen_found, self.debug = False, extra
                continue
            if self.source == "camera":
                self.screen_found = True
                self.blend_position(position)
            self.hazards = hazards
            rows = self.tracker.update(hazards, now, self.surfer.lanes)
            self.seen_at = now
            note = f"lane {self.surfer.lane}  {self.surfer.state.name.lower()}"
            self.debug = self.with_camera(perception.draw_debug(view, hazards, rows, self.surfer.lane, note), *extra)
            self.frames += 1
            if now - last_count_at >= 1.0:
                self.fps, self.frames, last_count_at = self.frames / (now - last_count_at), 0, now

    def blend_position(self, new: Position | None) -> None:
        old = self.position
        if new is None or old is None or time.monotonic() - self.seen_at > STALE_S:
            self.position = new
            return
        k = POSITION_BLEND
        self.position = Position(old.left + k * (new.left - old.left),
                                 old.distance + k * (new.distance - old.distance),
                                 old.facing_right_deg + k * (new.facing_right_deg - old.facing_right_deg))

    @staticmethod
    def with_camera(view: np.ndarray, wide: np.ndarray | None, corners: np.ndarray | None) -> np.ndarray:
        """Camera source: put the camera's picture, with the frame outlined, above the game view."""
        if wide is None:
            return view
        wide = wide.copy()
        cv2.polylines(wide, [corners.astype(np.int32)], True, (255, 255, 255), 2)
        wide = cv2.resize(wide, (view.shape[1], view.shape[1] * wide.shape[0] // wide.shape[1]))
        return np.vstack([wide, view])

    # ------------------------------------------------------------ act
    async def act(self) -> None:
        while True:
            await asyncio.sleep(1.0 / ACT_RATE)
            if not self.playing:
                continue
            now = time.monotonic()
            problem = self.robot.problem()
            if problem:
                self.stop(problem)
                continue
            if now - self.last_page_at > PAGE_TIMEOUT_S:
                self.stop("the game page went quiet")
                continue
            if self.game == "real":
                # The decisions were made in see_real; here the dog catches up with them.
                if self.surfer.free and self.body_queue:
                    self.surfer.begin(self.body_queue.pop(0), now)
                self.surfer.tick(now)
                continue
            fresh = now - self.seen_at <= STALE_S
            if fresh and self.surfer.free:
                self.tracker.predict(now)
                move, row = decide(self.tracker.rows, self.surfer.lane, self.tracker.speed, self.surfer.timing())
                if move != "run" and self.surfer.begin(move, now) and move in ("jump", "squat"):
                    row.handled = True  # one trick per row
            self.surfer.tick(now, self.position if fresh else None)

    # ------------------------------------------------------------ report
    async def report(self) -> None:
        while True:
            await asyncio.sleep(REPORT_S)
            if self.page is None:
                continue
            await self.send(self.status())
            if self.debug is not None:
                ok, jpeg = cv2.imencode(".jpg", self.debug, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    await self.send(jpeg.tobytes())


player: Player | None = None
ARGS = argparse.Namespace(sim=True, source="stream", hold=False, view_width=None, rotate=False, game="lane-hound", model=None)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global player
    robot = SimRobot() if ARGS.sim else RealRobot()
    finder = ScreenFinder(rotate_180=ARGS.rotate)
    if ARGS.view_width:
        finder.view_width_m = ARGS.view_width
    player = Player(robot, ARGS.source, ARGS.hold, finder, ARGS.game, ARGS.model)
    tasks = [asyncio.create_task(job()) for job in (player.see, player.act, player.report)]
    log.info("ready: %s robot, %s, eyes on the %s%s", robot.mode, ("the real game, learned eyes" if ARGS.model else "the real game") if ARGS.game == "real" else "Lane Hound",
             ARGS.source, ", holding lanes by camera" if ARGS.hold else "")
    yield
    player.stop("server shutting down")
    for task in tasks:
        task.cancel()


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def game_page(socket: WebSocket) -> None:
    await socket.accept()
    player.page = socket
    player.last_page_at = time.monotonic()
    try:
        while True:
            message = await socket.receive()
            if message["type"] == "websocket.disconnect":
                break
            player.last_page_at = time.monotonic()
            if message.get("bytes") is not None:
                player.stream_jpeg = message["bytes"]
                player.new_stream.set()
            elif message.get("text") is not None:
                await player.on_message(json.loads(message["text"]))
    except WebSocketDisconnect:
        pass
    finally:
        if player.page is socket:
            player.page = None
            player.stop("the game page closed")


@app.get("/status")
def status() -> dict:
    return player.status()


@app.post("/stop")
def stop() -> dict:
    player.stop("stopped with /stop")
    return player.status()


@app.post("/lie_down")
async def lie_down() -> dict:
    player.stop("lying down")
    ok, message = await player.robot.lie_down()
    return {"success": ok, "message": message}


@app.get("/view.jpg")
def view() -> Response:
    """What Pupper sees right now, with what it read from it drawn on top."""
    if player.debug is None:
        return Response(status_code=503, content="no picture yet")
    ok, jpeg = cv2.imencode(".jpg", player.debug)
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


class GameFiles(StaticFiles):
    """The game page. Never cached, so an edit to game/ shows up on reload."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


app.mount("/", GameFiles(directory=GAME_DIR, html=True))


def main() -> None:
    global ARGS
    parser = argparse.ArgumentParser(description="Pupper plays Lane Hound.")
    parser.add_argument("--sim", action="store_true", help="a pretend robot: run on a laptop, no ROS")
    parser.add_argument("--source", choices=["stream", "camera"], default="stream",
                        help="where Pupper's picture of the game comes from (default: stream)")
    parser.add_argument("--hold", action="store_true",
                        help="camera source: steer back to the lane using the position seen by the camera")
    parser.add_argument("--view-width", type=float, metavar="M",
                        help="camera source: width in metres of Pupper's view on the screen, green frame included")
    parser.add_argument("--rotate", action="store_true", help="camera source: the camera picture is upside down")
    parser.add_argument("--game", choices=["lane-hound", "real"], default="lane-hound",
                        help="real: the pictures are of Subway Surfers itself, sent by real/play.py")
    parser.add_argument("--model", metavar="FILE",
                        help="real game only: decide with this model (made by real/train.py) instead of the colour rules")
    parser.add_argument("--host", default="127.0.0.1", help="default: this machine only (use the SSH tunnel)")
    parser.add_argument("--port", type=int, default=8770)
    ARGS = parser.parse_args()
    if ARGS.model and ARGS.game != "real":
        parser.error("--model is for --game real")
    if ARGS.model and not Path(ARGS.model).is_file():
        parser.error(f"no model file at {ARGS.model}: make one with `python3 real/train.py` (see real/README.md), "
                     "and give the full path if the server is started from another folder")
    if ARGS.sim and ARGS.source == "camera":
        parser.error("--sim has no camera: use the stream source")
    if ARGS.game == "real" and ARGS.source == "camera":
        parser.error("the real game is too fine to read through Pupper's fisheye camera: use the stream source")
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning")


if __name__ == "__main__":
    main()
