"""Pupper finds its way by ear: the server that runs on the robot (or, with --sim, on a laptop).

    python3 echo/server.py --sim                # laptop: a pretend dog in a pretend room, two ears
    python3 echo/server.py --sim --ears four    # ... with four ears
    python3 echo/server.py --ears two           # Pupper: the compact USB-C receiver, two mics
    python3 echo/server.py --ears four          # Pupper: the main receiver, up to four mics

Then open the map: http://localhost:8771 (from a laptop, through
`ssh -N -L 8771:localhost:8771 pi@pupper<N>.local`).

It does two things at once (two asyncio tasks):

    act     20 times a second: navigator.tick() turns, listens, decides, walks
    report  5 times a second: tell the map page what Pupper hears and thinks

The page draws the map as Pupper builds it and has the buttons: Stand up,
Look, Explore, Stop; click the map to send Pupper there. With --sim the page
also shows the real room and the real dog, faintly, to compare.

Safety: Pupper only moves during a job (look / explore / goto). It stops by
itself if the map page closes or goes quiet, if it falls, or if the e-stop is
pressed; the joystick always outranks it. `curl -X POST localhost:8771/stop`
stops it too. It walks at most STEP_M between looks and keeps BODY_M from
everything it hears (navigator.py), but it only knows what echoes: soft things
(curtains, sofas, people's legs in loose trousers) are quiet. Stay close.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import logging
import math
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

import ears as ears_module
from dog import RealDog, SimDog
from navigator import WALK_SCALE, WALK_SPEED, Navigator
from sim_room import Post, SimRoom
from sound import SimSound, SoundCard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("echo")

PAGE_DIR = Path(__file__).resolve().parent / "page"
ACT_RATE = 20.0        # Hz: the speed commands (cmd_vel_mux drops a source quiet for 0.5 s)
REPORT_S = 0.2         # how often the page hears from us
PAGE_TIMEOUT_S = 2.0   # the page says hello twice a second; quiet this long while moving: stop

# The pretend rooms (--room), and where the pretend dog starts in each: (x, y, heading deg).
SIM_ROOMS = {
    "room": (lambda: SimRoom.box(4.0, 3.0, seed=1), (1.2, 1.0, 20.0)),
    "arena": (lambda: SimRoom.box(2.4, 2.0, clutter=0.5, ceiling_m=None, seed=1), (0.8, 0.8, -30.0)),
    "slanted": (lambda: SimRoom.polygon([(0, 0), (4.2, 0), (3.4, 3.0), (0, 2.6)], seed=1), (1.0, 1.0, 70.0)),
    "post": (lambda: SimRoom(SimRoom.box(4.5, 3.5).walls, posts=[Post((2.4, 1.5), 0.06)], seed=1), (1.1, 1.3, 10.0)),
}


class Session:
    """Everything the server holds: the dog, its ears, the navigator, and what to tell the page."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.geometry = ears_module.load_geometry(args.ears)
        self.lines: collections.deque[str] = collections.deque(maxlen=80)
        self.page_heard = 0.0
        self.pages = 0
        self.truth: dict | None = None
        if args.sim:
            make_room, start = SIM_ROOMS[args.room]
            self.room = make_room()
            self.dog = SimDog(*start)
            self.start = start
            # The pretend dog's ears are not quite where ears.yaml says: a ruler is good to a few millimetres.
            nudge = np.random.default_rng(7)
            flat = [1.0, 1.0, 0.0]
            self.sound = SimSound(self.room, self.dog,
                                  self.geometry.speaker + 0.005 * nudge.standard_normal(3) * flat,
                                  self.geometry.mics + 0.005 * nudge.standard_normal(self.geometry.mics.shape) * flat)
        else:
            self.dog = RealDog()
            self.sound = SoundCard(len(self.geometry.mics), mic=args.mic, speaker=args.speaker)
        self.navigator = Navigator(self.dog, self.sound, ears_module.Ears(self.geometry), say=self.say)
        self.say(f"{args.ears} ears ({', '.join(self.geometry.names)}), sound: {self.sound.name}")

    def say(self, line: str) -> None:
        log.info(line)
        self.lines.append(f"{time.strftime('%H:%M:%S')}  {line}")

    # ---------------------------------------------------------------- what the page asks for
    async def do(self, message: dict) -> None:
        self.page_heard = time.monotonic()
        what = message.get("do")
        now = time.monotonic()
        nav = self.navigator
        if what == "hello":
            return
        if what == "stop":
            nav.stop("stopped")
        elif what == "stand":
            nav.stop()
            self.say("standing up ...")
            ok, text = await self.dog.stand_up()
            self.say(text if ok else f"could not stand up: {text}")
        elif what == "lie":
            nav.stop()
            ok, text = await self.dog.lie_down()
            self.say(text if ok else f"could not lie down: {text}")
        elif what == "forget":
            nav.forget()
        elif what in ("look", "explore", "goto", "walk_test"):
            if nav.busy:
                return self.say(f"busy ({nav.message}): press Stop first")
            if what == "walk_test":
                refused = nav.walk_test(now, float(message.get("m", 1.0)))
            else:
                goal = (float(message["x"]), float(message["y"])) if what == "goto" else None
                refused = nav.start(what, now, goal)
            if refused:
                self.say(refused)

    # ---------------------------------------------------------------- the two tasks
    async def act(self) -> None:
        period = 1.0 / ACT_RATE
        while True:
            now = time.monotonic()
            nav = self.navigator
            if nav.busy and now - self.page_heard > PAGE_TIMEOUT_S:
                nav.stop("stopped: the map page went quiet (closed, or the connection dropped)")
            try:
                nav.tick(now)
            except Exception:  # a bug must not leave the dog walking
                log.exception("tick failed")
                nav.stop("stopped: internal error (see the log)")
            await asyncio.sleep(period)

    def report(self) -> dict:
        report = self.navigator.snapshot()
        report["log"] = list(self.lines)[-14:]
        report["robot"] = {"mode": self.dog.mode, "problem": self.dog.problem()}
        if self.args.sim:
            report["truth"] = self.sim_truth()
        return report

    def sim_truth(self) -> dict:
        """The real room and the real dog, in the map's frame (the dog's starting pose is the origin)."""
        x0, y0, heading0 = self.start
        c, s = math.cos(math.radians(heading0)), math.sin(math.radians(heading0))

        def to_map(p) -> list[float]:
            dx, dy = p[0] - x0, p[1] - y0
            return [c * dx + s * dy, -s * dx + c * dy]

        return {
            "walls": [[to_map(w.a), to_map(w.b)] for w in self.room.walls],
            "posts": [{"at": to_map(p.at), "radius": p.radius} for p in self.room.posts],
            "pose": {"at": to_map((self.dog.x, self.dog.y)), "heading_deg": math.degrees(self.dog.heading) - heading0},
        }


session: Session | None = None
ARGS: argparse.Namespace | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global session
    session = Session(ARGS)   # here, not at import: the real robot needs the running event loop
    task = asyncio.create_task(session.act())
    yield
    session.navigator.stop("server stopping")
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def map_page(socket: WebSocket) -> None:
    await socket.accept()
    session.pages += 1

    async def listen() -> None:
        while True:
            await session.do(await socket.receive_json())

    listener = asyncio.create_task(listen())
    try:
        while not listener.done():
            await socket.send_json(session.report())
            await asyncio.sleep(REPORT_S)
        listener.result()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        listener.cancel()
        session.pages -= 1
        if session.pages == 0 and session.navigator.busy:
            session.navigator.stop("stopped: the map page closed")


@app.get("/status")
def status() -> dict:
    report = session.report()
    report.pop("profiles", None)
    return report


@app.post("/stop")
def stop() -> dict:
    session.navigator.stop("stopped (POST /stop)")
    return {"ok": True}


@app.post("/lie_down")
async def lie_down() -> dict:
    session.navigator.stop()
    ok, text = await session.dog.lie_down()
    return {"ok": ok, "message": text}


class NoCache(StaticFiles):
    """The page's files, never cached: edit, reload, see it."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


app.mount("/", NoCache(directory=PAGE_DIR, html=True), name="page")


def main() -> None:
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sim", action="store_true", help="a pretend dog in a pretend room (no robot, no microphones)")
    parser.add_argument("--ears", choices=("two", "four"), default="two", help="which set of ears in ears.yaml")
    parser.add_argument("--room", choices=sorted(SIM_ROOMS), default="room", help="--sim: which pretend room")
    parser.add_argument("--mic", default="DJI", help="recording device: a number or part of its name (mic_check.py --list)")
    parser.add_argument("--speaker", default=None, help="playing device: a number or part of its name (default: the system's)")
    parser.add_argument("--port", type=int, default=8771)
    ARGS = parser.parse_args()
    log.info("walking: %.2f m/s commanded, x%.2f", WALK_SPEED, WALK_SCALE)
    # localhost only: from a laptop, reach it through the ssh port forward.
    uvicorn.run(app, host="127.0.0.1", port=ARGS.port, log_level="warning")


if __name__ == "__main__":
    main()
