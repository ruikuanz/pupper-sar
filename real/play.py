"""The real Subway Surfers, twice on one screen: a person plays the left one, Pupper the right one.

    python3 real/play.py                 # the two games + the control strip, in slow motion
    python3 real/play.py --speed 1       # ... at the game's own speed
    python3 real/play.py --no-pupper     # the two games and the webcam player only
    python3 real/play.py --record        # ... and save the person's play, for real/train.py

Nothing of the game is copied or rebuilt here. This program opens the game's
own web page (GAME_URL: the browser release, on the publisher's portal) in two
Chrome windows, side by side, and works them the way two players would:

    left window   the HUMAN's game. Arrow keys, or the webcam: step left or
                  right, jump, squat (game/js/pose_moves.js, the same reader
                  Lane Hound uses). The moves become arrow keys in this window.
    right window  PUPPER's game. Its picture is sent to surfer/server.py
                  (--game real), 30 times a second. Pupper decides, the key
                  goes to this window, and the dog makes the same move.
    bottom strip  the controls: start both runs, let Pupper play, the webcam.

Why two windows and not one page with two frames: the game's page does not
allow itself to be put inside another page, and it should not be worked around.

The windows belong to their own Chrome profile (PROFILE_DIR), so the game's
progress here is separate from your everyday browser. The first time, each
window shows the game's tutorial: play it through once by hand (arrow keys).

Keys reach each window through Chrome's remote-control connection, not through
the real keyboard, so both games can be played at the same moment whichever
window has the focus.

Slow motion. Pupper needs 0.8 s to shuffle sideways and over a second to jump;
the game's runner does those in 0.3 s and 0.7 s. So both games run slowed down
(--speed, default RUN_SPEED; the strip has a slider): at 0.35 the runner's
moves take as long as the dog's. Only the clock the game reads is slowed
(real/clock.js); the game itself is untouched, and both players get the same
speed, so it is still a fair race. The clock is only slow while a run is on
screen: menus and the "Save me!" countdown go by at normal speed.

Recording. With "Record my play" on, every picture of the human's run and
every arrow key (or webcam move) is saved under real/recordings/. That is the
teaching material for Pupper's learned eyes: real/train.py turns it into
real/model.npz, and `surfer/server.py --game real --model real/model.npz`
plays from it.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import http.server
import json
import logging
import shutil
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import websockets
from playwright.async_api import BrowserContext, Page, async_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "surfer"))
import real_perception  # noqa: E402  (which screen is showing: a run, the menu, the score)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("real")

GAME_URL = "https://poki.com/en/g/subway-surfers"
GAME_ELEMENT = "#game-element"            # the frame on that page that holds the game
PROFILE_DIR = Path(__file__).resolve().parent / ".chrome-profile"
PANEL_PORT = 8780                         # the control strip is served from here (game/real.html)
PANEL_HEIGHT = 330                        # px of the screen the control strip's window takes (its title and address bars included)
STREAM_FPS = 30                           # pictures a second sent to Pupper
STREAM_QUALITY = 80                       # JPEG quality of those pictures
HEARTBEAT_S = 0.25                        # the server stops Pupper if it hears nothing for 1.5 s
KEYS = {"left": "ArrowLeft", "right": "ArrowRight", "jump": "ArrowUp", "squat": "ArrowDown"}
MOVE_OF_KEY = {key: move for move, key in KEYS.items()}
# --- slow motion ------------------------------------------------------------------
CLOCK_SCRIPT = Path(__file__).resolve().parent / "clock.js"
RUN_SPEED = 0.35                          # how fast a run plays (1 = the game's own speed)
SPEED_MIN, SPEED_MAX = 0.15, 1.0
RUN_OVER_S = 0.7                          # no run on screen for this long (real s): back to normal speed
IDLE_LOOK_EVERY = 3                       # the human's window, when not recording: look at every 3rd picture
# --- recording a person's play ----------------------------------------------------
RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
RECORD_SIZE = (320, 180)                  # px of the saved pictures (the model uses half of this)
RECORD_QUALITY = 88
RECORD_EVERY_S = 1 / 40                   # at most this many pictures per second of GAME time
RECORD_MIN_RUN_S = 2.0                    # a run shorter than this (game s) is not kept
# Where to click, as a fraction of the game's picture.
CLICK_PLAY = (371 / 640, 327 / 360)       # the PLAY button of the score screen
CLICK_MENU = (0.5, 0.55)                  # anywhere on the title screen starts a run


class GameWindow:
    """One Chrome window with the game in it."""

    def __init__(self, name: str, page: Page) -> None:
        self.name = name
        self.page = page
        self.box: dict | None = None      # where the game is on the page (CSS px)
        self.cdp = None
        self.jpeg: bytes | None = None    # the newest screencast picture of the whole page
        self.count = 0                    # pictures so far
        self.fresh = asyncio.Event()      # set when a new picture has arrived
        self.viewport = 1.0               # CSS px width of the window's page
        self.running = False              # is a run on screen?
        self.last_run_at = 0.0
        self.clock = 1.0                  # how fast this game's clock is running right now

    async def open(self, bounds: dict) -> None:
        self.cdp = await self.page.context.new_cdp_session(self.page)
        window = await self.cdp.send("Browser.getWindowForTarget")
        await self.cdp.send("Browser.setWindowBounds", {"windowId": window["windowId"], "bounds": bounds})
        # Each window believes it has the focus, so the game in it never pauses
        # because the other window was clicked.
        await self.cdp.send("Emulation.setFocusEmulationEnabled", {"enabled": True})
        await self.page.goto(GAME_URL, wait_until="domcontentloaded")
        await self.page.locator(GAME_ELEMENT).wait_for(timeout=60000)
        await self.find_game()
        self.cdp.on("Page.screencastFrame", lambda event: asyncio.ensure_future(self.on_frame(event)))
        await self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 85, "everyNthFrame": 2})

    async def find_game(self) -> None:
        self.box = await self.page.locator(GAME_ELEMENT).bounding_box()
        self.viewport = await self.page.evaluate("window.innerWidth")

    async def on_frame(self, event: dict) -> None:
        self.jpeg = base64.b64decode(event["data"])
        self.count += 1
        self.fresh.set()
        await self.cdp.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})

    async def set_clock(self, speed: float) -> None:
        """Run this game's clock at `speed` from now on (real/clock.js is in every frame of the page)."""
        self.clock = speed
        for frame in self.page.frames:
            try:
                await frame.evaluate("(speed) => window.__gameClock && window.__gameClock.set(speed)", speed)
            except Exception:  # a frame that is loading or just went away (an advert): nothing to set
                pass

    def latest(self) -> np.ndarray | None:
        """The game's newest picture, 640 x 360, cut out of the newest picture of the page."""
        if self.jpeg is None:
            return None
        page_picture = cv2.imdecode(np.frombuffer(self.jpeg, np.uint8), cv2.IMREAD_COLOR)
        return self.crop(page_picture) if page_picture is not None else None

    def crop(self, page_picture: np.ndarray) -> np.ndarray | None:
        """The game's part of a picture of the whole page, as 640 x 360."""
        if self.box is None:
            return None
        scale = page_picture.shape[1] / self.viewport
        x0, y0 = int(self.box["x"] * scale), int(self.box["y"] * scale)
        x1, y1 = int((self.box["x"] + self.box["width"]) * scale), int((self.box["y"] + self.box["height"]) * scale)
        part = page_picture[max(y0, 0):y1, max(x0, 0):x1]
        if part.size == 0:
            return None
        return cv2.resize(part, (real_perception.VIEW_W, real_perception.VIEW_H), interpolation=cv2.INTER_AREA)

    async def view(self) -> np.ndarray | None:
        """The game's picture right now (one screenshot: for now-and-then use)."""
        await self.find_game()
        if self.box is None:
            return None
        png = await self.page.screenshot(clip=self.box, type="jpeg", quality=85)
        picture = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        return cv2.resize(picture, (real_perception.VIEW_W, real_perception.VIEW_H), interpolation=cv2.INTER_AREA)

    async def click(self, where: tuple[float, float]) -> None:
        await self.page.mouse.click(self.box["x"] + where[0] * self.box["width"], self.box["y"] + where[1] * self.box["height"])

    async def press(self, move: str) -> None:
        if move in KEYS:
            await self.page.keyboard.press(KEYS[move])

    async def start_run(self) -> str:
        """Get this game from its title or score screen into a run. Returns what it did."""
        picture = await self.view()
        if picture is None:
            return "can't see the game"
        state = real_perception.screen(picture)
        if state == "run":
            # A click inside the game gives its frame the keyboard; it does nothing else during a run.
            await self.click((0.5, 0.75))
            return "already running"
        if state == "save_me":
            return 'showing "Save me!": wait for the score screen'   # its buttons spend keys or play an advert
        await self.click(CLICK_PLAY if state == "score" else CLICK_MENU)
        return "started"


class Recorder:
    """Saves a person's play: the pictures of each run, and when which key was pressed.

        real/recordings/<day-time>/run03/00000.jpg, 00001.jpg, ...   the pictures
        real/recordings/<day-time>/run03/run.json                    when each was taken, and the keys

    All times are GAME seconds since the run started, so a recording made in
    slow motion teaches the same lesson as one made at full speed. A run ends
    when the game stops showing one (a crash); real/train.py leaves out the
    last second of every run, which is the mistake that ended it.
    """

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.on = False
        self.session: Path | None = None
        self.run: Path | None = None
        self.now = 0.0               # game seconds, counted from the pictures
        self.seen_at: float | None = None
        self.clock = 1.0
        self.started = 0.0           # game time when this run started
        self.saved_at = -1.0
        self.last_run_at = 0.0       # real time a run was last on screen
        self.times: list[float] = []
        self.keys: list[list] = []
        self.kept_runs, self.kept_seconds, self.kept_keys = 0, 0.0, 0

    def add(self, view: np.ndarray, running: bool, clock: float, real_now: float) -> None:
        """One picture of the human's game."""
        if self.seen_at is not None:
            self.now += (real_now - self.seen_at) * self.clock
        self.seen_at, self.clock = real_now, clock
        if not self.on:
            self.close()
            return
        if running:
            self.last_run_at = real_now
            if self.run is None:
                self.open()
            if self.now - self.saved_at >= RECORD_EVERY_S:
                self.saved_at = self.now
                small = cv2.resize(view, RECORD_SIZE, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(self.run / f"{len(self.times):05d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, RECORD_QUALITY])
                self.times.append(round(self.now - self.started, 4))
        elif self.run is not None and real_now - self.last_run_at > RUN_OVER_S:
            self.close()

    def key(self, move: str, real_now: float) -> None:
        """The person pressed a key (or made a move in front of the webcam)."""
        if self.run is None or self.seen_at is None:
            return
        at = self.now + (real_now - self.seen_at) * self.clock - self.started
        self.keys.append([round(at, 4), move])

    def open(self) -> None:
        if self.session is None:
            self.session = self.folder / time.strftime("%Y%m%d-%H%M%S")
        number = 0
        while (self.session / f"run{number:02d}").exists():
            number += 1
        self.run = self.session / f"run{number:02d}"
        self.run.mkdir(parents=True)
        self.started, self.saved_at, self.times, self.keys = self.now, -1.0, [], []

    def close(self) -> None:
        """The run is over: write its list of times and keys, or throw away a run too short to teach anything."""
        if self.run is None:
            return
        run, self.run = self.run, None
        seconds = self.times[-1] if self.times else 0.0
        if seconds < RECORD_MIN_RUN_S:
            shutil.rmtree(run, ignore_errors=True)
            return
        keys = [key for key in self.keys if key[0] <= seconds]
        (run / "run.json").write_text(json.dumps({
            "size": list(RECORD_SIZE), "seconds": seconds, "clock": self.clock, "times": self.times, "keys": keys}))
        self.kept_runs, self.kept_seconds, self.kept_keys = self.kept_runs + 1, self.kept_seconds + seconds, self.kept_keys + len(keys)
        log.info("recorded %s: %.1f game s, %d pictures, %d keys", run.name, seconds, len(self.times), len(keys))

    def summary(self) -> str:
        kept = f"{self.kept_runs} run{'' if self.kept_runs == 1 else 's'} saved, {self.kept_seconds / 60:.1f} game minutes, {self.kept_keys} moves"
        if not self.on:
            return kept if self.kept_runs else "not recording"
        return f"recording{' a run' if self.run is not None else ' (waiting for a run)'}; {kept}"


class Bridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.human: GameWindow | None = None
        self.pupper: GameWindow | None = None
        self.panel: Page | None = None
        self.socket = None               # to surfer/server.py
        self.sent = 0
        self.speed = min(SPEED_MAX, max(SPEED_MIN, args.speed))   # how fast a run plays in both windows
        self.recorder = Recorder(RECORDINGS_DIR)
        self.recorder.on = args.record

    # ------------------------------------------------------------ the control strip
    async def to_panel(self, function: str, value) -> None:
        if self.panel is None or self.panel.is_closed():
            return
        try:
            await self.panel.evaluate(f"(value) => window.panel && window.panel.{function}(value)", value)
        except Exception:  # the strip is reloading or closing: nothing to show it to
            pass

    async def from_panel(self, action: str, value=None) -> str:
        """A button on the control strip, or a move from the webcam."""
        if action == "human":
            await self.human.press(value)
            return "ok"
        if action == "settings":
            return json.dumps({"speed": self.speed, "recording": self.recorder.on})
        if action == "speed":
            self.speed = min(SPEED_MAX, max(SPEED_MIN, float(value)))
            return f"{self.speed:.2f}"
        if action == "record":
            self.recorder.on = bool(value)
            if not self.recorder.on:
                self.recorder.close()
            return self.recorder.summary()
        if action == "start_runs":
            results = await asyncio.gather(self.human.start_run(), self.pupper.start_run())
            return f"human: {results[0]}; Pupper: {results[1]}"
        if action in ("start", "stop", "stand_up", "lie_down"):
            if self.socket is None:
                return "not connected to surfer/server.py"
            await self.socket.send(json.dumps({"type": "control", "action": action}))
            return "ok"
        return f"unknown: {action}"

    # ------------------------------------------------------------ Pupper's link
    def on_key(self, source: dict, key: str) -> None:
        """An arrow key went down in one of the windows (real/clock.js reports it)."""
        if self.human is not None and source.get("page") is self.human.page and key in MOVE_OF_KEY:
            self.recorder.key(MOVE_OF_KEY[key], time.monotonic())

    async def watch(self, window: GameWindow) -> None:
        """Look at each new picture of one game: is a run on, how fast should its clock go,
        and then Pupper's picture goes to the server and the human's to the recorder."""
        looked = 0
        while True:
            await window.fresh.wait()
            window.fresh.clear()
            looked += 1
            try:
                await self.look(window, looked)
            except Exception as error:  # a window that is reloading or closing: try the next picture
                log.debug("%s: %s", window.name, error)

    async def look(self, window: GameWindow, looked: int) -> None:
        is_human = window is self.human
        if is_human and not self.recorder.on and looked % IDLE_LOOK_EVERY:
            return
        view = window.latest()
        if view is None:
            return
        now = time.monotonic()
        if real_perception.screen(view) == "run":
            window.running, window.last_run_at = True, now
        elif window.running and now - window.last_run_at > RUN_OVER_S:
            window.running = False
        # Slow motion only while a run is on screen.
        wanted = self.speed if window.running else 1.0
        if abs(wanted - window.clock) > 1e-6:
            await window.set_clock(wanted)
            if not is_human:
                await self.tell_server()
        if is_human:
            self.recorder.add(view, window.running, window.clock, now)
        elif self.socket is not None:
            ok, jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
            try:
                await self.socket.send(jpeg.tobytes())
                self.sent += 1
            except websockets.ConnectionClosed:
                pass

    async def tell_server(self) -> None:
        """Tell the server the game is still here, and how fast its clock is running."""
        if self.socket is None:
            return
        try:
            await self.socket.send(json.dumps({"type": "state", "game": "real", "clock": self.pupper.clock}))
        except websockets.ConnectionClosed:
            pass

    async def heartbeat(self) -> None:
        """Four times a second: the server hears from us. Every 2 s: where the games are on
        their pages (a resized window moves them), and their clocks are set again (a page
        that reloaded itself starts at the speed it was opened with)."""
        ticks = 0
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            ticks += 1
            await self.tell_server()
            if ticks % 4 == 0:
                await self.to_panel("recorder", self.recorder.summary())
            if ticks % 8 == 0:
                for window in (self.human, self.pupper):
                    if window is not None:
                        try:
                            await window.find_game()
                            await window.set_clock(window.clock)
                        except Exception:  # the window is navigating or closing
                            pass

    async def link(self) -> None:
        """Stay connected to surfer/server.py: its moves become keys in Pupper's window."""
        url = self.args.server.replace("http", "ws", 1).rstrip("/") + "/ws"
        while True:
            try:
                async with websockets.connect(url, max_size=None) as socket:
                    self.socket = socket
                    log.info("connected to Pupper's server at %s", url)
                    await self.to_panel("link", True)
                    async for message in socket:
                        if isinstance(message, bytes):   # what Pupper sees, with what it read drawn on top
                            await self.to_panel("eye", base64.b64encode(message).decode())
                            continue
                        message = json.loads(message)
                        if message.get("type") == "move":
                            await self.pupper.press(message["move"])
                        elif message.get("type") in ("status", "ready", "body"):
                            await self.to_panel("message", message)
            except (OSError, websockets.WebSocketException):
                pass
            if self.socket is not None:
                log.info("lost Pupper's server; trying again")
            self.socket = None
            await self.to_panel("link", False)
            await asyncio.sleep(1.0)

    # ------------------------------------------------------------ the windows
    async def new_window(self, context: BrowserContext, opener: Page) -> Page:
        """A page in a window of its own (a context's pages otherwise open as tabs)."""
        cdp = await context.new_cdp_session(opener)
        async with context.expect_page() as arriving:
            await cdp.send("Target.createTarget", {"url": "about:blank", "newWindow": True})
        return await arriving.value

    async def run(self) -> None:
        serve_panel()
        async with async_playwright() as playwright:
            context = await playwright.chromium.launch_persistent_context(
                str(PROFILE_DIR), channel="chrome", headless=self.args.headless, no_viewport=True,
                ignore_default_args=["--enable-automation"],
                args=["--autoplay-policy=no-user-gesture-required"] + (["--mute-audio"] if self.args.mute else []),
            )
            panel_url = f"http://localhost:{PANEL_PORT}"
            await context.grant_permissions(["camera"], origin=panel_url)
            # Before any page loads: the game's clock (it starts at normal speed and is slowed
            # when a run is seen) and the report of the arrow keys.
            await context.add_init_script(CLOCK_SCRIPT.read_text().replace("__SPEED__", "1"))
            await context.expose_binding("gameKey", self.on_key)
            first = context.pages[0] if context.pages else await context.new_page()
            width, height = self.args.screen or await first.evaluate("[screen.availWidth, screen.availHeight]")
            top = await first.evaluate("screen.availTop || 0")
            games_height = height - PANEL_HEIGHT
            half = width // 2

            # Human on the left, Pupper on the right.
            self.human = GameWindow("human", first)
            self.pupper = GameWindow("pupper", await self.new_window(context, first))
            self.panel = await self.new_window(context, first)
            await asyncio.gather(
                self.human.open({"left": 0, "top": top, "width": half, "height": games_height, "windowState": "normal"}),
                self.pupper.open({"left": half, "top": top, "width": width - half, "height": games_height, "windowState": "normal"}),
            )

            await self.panel.expose_function("bridge", self.from_panel)
            cdp = await context.new_cdp_session(self.panel)
            window = await cdp.send("Browser.getWindowForTarget")
            await cdp.send("Browser.setWindowBounds", {"windowId": window["windowId"], "bounds": {
                "left": 0, "top": top + games_height, "width": width, "height": PANEL_HEIGHT, "windowState": "normal"}})
            await self.panel.goto(f"{panel_url}/real.html" + ("?pupper=0" if self.args.no_pupper else ""))
            log.info("ready: the human's game is on the left, Pupper's on the right; runs play at %.2f of full speed", self.speed)

            jobs = [asyncio.create_task(self.heartbeat()), asyncio.create_task(self.watch(self.human)),
                    asyncio.create_task(self.watch(self.pupper))]
            if not self.args.no_pupper:
                jobs.append(asyncio.create_task(self.link()))
            closed = asyncio.Event()
            context.on("close", lambda _context: closed.set())
            try:
                if self.args.seconds:
                    await asyncio.wait_for(closed.wait(), timeout=self.args.seconds)
                else:
                    await closed.wait()   # until the windows are closed (or Ctrl-C)
            except asyncio.TimeoutError:
                pass
            finally:
                for job in jobs:
                    job.cancel()
                self.recorder.close()
                if not closed.is_set():
                    await context.close()


def serve_panel() -> None:
    """Serve game/ (the control strip and the webcam code) on localhost, where the browser allows the camera."""

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            pass

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

    handler = functools.partial(Quiet, directory=str(ROOT / "game"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PANEL_PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="The real Subway Surfers: a person on the left, Pupper on the right.")
    parser.add_argument("--server", default="http://localhost:8770", help="surfer/server.py --game real (default: %(default)s)")
    parser.add_argument("--no-pupper", action="store_true", help="only the two games and the webcam player")
    parser.add_argument("--speed", type=float, default=RUN_SPEED,
                        help="how fast runs play in both windows, 1 = the game's own speed (default: %(default)s)")
    parser.add_argument("--record", action="store_true", help="save the human's play under real/recordings/ from the start")
    parser.add_argument("--mute", action="store_true", help="no sound from either game (in slow motion the sound is not slowed)")
    parser.add_argument("--screen", type=int, nargs=2, metavar=("W", "H"), help="lay the windows out for a screen this big (default: this one)")
    parser.add_argument("--headless", action="store_true", help="no windows on screen (for testing)")
    parser.add_argument("--seconds", type=float, help="close by itself after this long (for testing)")
    args = parser.parse_args()
    try:
        asyncio.run(Bridge(args).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
