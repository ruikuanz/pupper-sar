"""The assistant's robot server: HTTP + WebSocket API on port 8000, and the web app.

Run on the Pupper after the robot stack is up (gemini/start.sh does both):
    python3 gemini/server.py
On a laptop, with a pretend robot (the desk tests' fake ROS, in real time):
    python3 gemini/server.py --sim          then open http://localhost:8000

The web app gets Gemini's instructions and tools from GET /tools (gemini/prompt.md
and the @tool functions in gemini/tools/) and sends every tool call to POST /tools/<name>. Every call
returns {"success": bool, "message": str}, which goes straight back to Gemini,
so Gemini hears about refused speeds, unknown tricks and failed controller
switches. The other endpoints are for the debug panel and for testing by hand.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import psutil
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

SIM = "--sim" in sys.argv
if SIM:  # a pretend ROS has to be in place before robot_api is imported
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    import test_assistant as pretend  # noqa: E402

import tool_registry as gemini_tools
from fisheye_converter import equirect_height, fisheye_to_equirectangular, load_camera_model
from robot_api import LIMITS, Robot

behaviors = gemini_tools.load_tools()  # gemini/tools/: every tool, and the behaviors they call

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("server")

HERE = Path(__file__).resolve().parent
WEB_DIST = HERE / "web" / "dist"
PROMPT_FILE = HERE / "prompt.md"
# The platform's battery reader, if this robot has it (it ships with the pupperv3 software).
BATTERY_SCRIPT = Path.home() / "pupperv3-monorepo" / "robot" / "utils" / "check_batt_voltage.py"

tools: Robot | None = None


class Detections:
    """Forwards person detections (ZMQ on localhost:5556) to browser WebSockets.

    The follower publishes them while it follows (it keeps the Hailo off
    otherwise); the monorepo detector, when used instead, publishes them always.
    """

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def run(self) -> None:
        try:
            import zmq
            import zmq.asyncio
        except ImportError:
            log.warning("pyzmq missing; eye tracking from robot detections disabled")
            return
        sock = zmq.asyncio.Context().socket(zmq.SUB)
        sock.connect("tcp://127.0.0.1:5556")
        sock.subscribe(b"")
        while True:
            payload = (await sock.recv_multipart())[-1].decode("utf-8", errors="ignore")
            if payload.startswith("{"):
                for ws in list(self.clients):
                    try:
                        await ws.send_text(payload)
                    except Exception:
                        self.clients.discard(ws)


detections = Detections()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tools
    if SIM:
        start_pretend_robot()
    tools = Robot()
    task = asyncio.create_task(detections.run())
    yield
    task.cancel()


app = FastAPI(title="Pupper assistant robot server", lifespan=lifespan)
# The web app is served from this same origin; also allow a dev server on the laptop.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


def T() -> Robot:
    if tools is None:
        raise HTTPException(503, "tool server not ready")
    return tools


# ------------------------------------------------------------------ models
class MoveVelocity(BaseModel):
    forward_backward_velocity: float = Field(..., description="m/s, positive = forward")
    right_left_velocity: float = Field(..., description="m/s, positive = right")
    turning_velocity: float = Field(..., description="deg/s, positive = turn right")
    duration: float = Field(..., description="seconds")


class MoveDirection(BaseModel):
    heading: float = Field(..., description="degrees, 0 = forward, 90 = right")
    speed: float
    duration: float


class Trick(BaseModel):
    animation_name: str


class TrickAction(BaseModel):
    action: str


class Wait(BaseModel):
    duration: float


class Volume(BaseModel):
    volume: int = Field(..., ge=0, le=150)


# ------------------------------------------------------------------ Gemini's tools
def system_prompt() -> str:
    """gemini/prompt.md without its <!-- comments -->."""
    text = PROMPT_FILE.read_text() if PROMPT_FILE.exists() else ""
    return re.sub(r"<!--.*?-->", "", text, flags=re.S).strip()


@app.get("/tools")
def tool_list():
    """What Gemini is told: the system prompt and the tool declarations."""
    return {"prompt": system_prompt(), "tools": gemini_tools.declarations(T())}


@app.post("/tools/{name}")
async def tool_call(name: str, args: dict = Body(default={})):
    res = await gemini_tools.call(T(), name, args)
    log.info("tool %s(%s) -> %s: %s", name, args, "ok" if res.ok else "FAILED", res.message)
    return res.as_dict()


# ------------------------------------------------------------------ commands
@app.post("/walking/activate")
async def walking_activate():
    return (await behaviors.stand_up(T())).as_dict()


@app.post("/walking/deactivate")
async def walking_deactivate():
    return (await behaviors.lie_down(T())).as_dict()


@app.post("/move/velocity")
async def move_velocity(req: MoveVelocity):
    # The browser uses right/clockwise-positive; ROS uses left/counter-clockwise-positive.
    res = await behaviors.walk(T(), req.forward_backward_velocity, -req.right_left_velocity, -req.turning_velocity, req.duration)
    return res.as_dict()


@app.post("/move/direction")
async def move_direction(req: MoveDirection):
    turn = await behaviors.walk(T(), 0.0, 0.0, -90.0 if req.heading > 0 else 90.0, abs(req.heading) / 90.0) if req.heading else None
    if turn and not turn.ok:
        return turn.as_dict()
    return (await behaviors.walk(T(), req.speed, 0.0, 0.0, req.duration)).as_dict()


@app.post("/animation")
async def animation(req: Trick):
    return (await behaviors.play_trick(T(), req.animation_name)).as_dict()


@app.post("/interactive_action")
async def interactive_action(req: TrickAction):
    return (await behaviors.interactive_action(T(), req.action)).as_dict()


@app.get("/interactive_actions")
def interactive_actions():
    """Actions of interactive tricks (e.g. shake a paw while sitting): id -> trick and description."""
    return {aid: {"trick": t.name, "description": t.actions[clip]} for aid, (t, clip) in T().action_ids().items()}


@app.get("/animations")
def animations():
    return available_tricks()


def available_tricks() -> dict:
    """The tricks in <repo>/tricks, every one played by a policy."""
    return {
        name: {"description": t.description, "duration_s": round(t.total_time, 1), "kind": t.kind}
        for name, t in T().tricks.items()
    }


@app.post("/queue/wait")
async def queue_wait(req: Wait):
    return (await T().wait(req.duration)).as_dict()


@app.post("/queue/reset")
async def queue_reset():
    return (await T().clear_queue()).as_dict()


@app.post("/queue/stop")
@app.post("/stop/immediate")
async def stop_immediate():
    return (await T().immediate_stop(relax_playing_trick=True)).as_dict()


@app.get("/queue/status")
def queue_status():
    return T().status()


@app.post("/following/activate")
async def following_on():
    return (await T().set_following(True)).as_dict()


@app.post("/following/deactivate")
async def following_off():
    return (await T().set_following(False)).as_dict()


@app.get("/system/mode")
async def system_mode():
    return (await T().mode()).as_dict()


@app.get("/limits")
def limits():
    return LIMITS.__dict__


# ------------------------------------------------------------------ sensing
# What Gemini sees: the fisheye frame undistorted to an equirectangular view (the
# double sphere lens model, a 220 degree field of view, 700 pixels wide), so
# people look like people instead of bending around the edge of the lens.
CAMERA_PARAMS = HERE.parent / "pupper" / "camera_params.yaml"
EQUIRECT_WIDTH = 700
EQUIRECT_FOV_DEG = 220.0
camera_model = None


def undistort_jpeg(data: bytes) -> bytes | None:
    global camera_model
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if camera_model is None or (camera_model.width, camera_model.height) != (w, h):
        camera_model = load_camera_model(CAMERA_PARAMS, w, h)
    equirect = fisheye_to_equirectangular(frame, camera_model, EQUIRECT_WIDTH,
                                          equirect_height(EQUIRECT_WIDTH, EQUIRECT_FOV_DEG),
                                          h_fov_deg=EQUIRECT_FOV_DEG)
    ok, jpeg = cv2.imencode(".jpg", equirect, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg.tobytes() if ok else None


@app.get("/camera/image")
def camera_image():
    try:
        msg = T().latest_image.get_nowait()
    except Exception:
        return Response(status_code=503, content="no camera frame yet")
    T().latest_image.put_nowait(msg)  # keep it for the next poll
    jpeg = undistort_jpeg(bytes(msg.data))
    if jpeg is None:
        return Response(status_code=503, content="could not decode the camera frame")
    return Response(content=jpeg, media_type="image/jpeg")


@app.websocket("/ws/detections")
async def ws_detections(ws: WebSocket):
    await ws.accept()
    detections.clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        detections.clients.discard(ws)


@app.get("/system/status")
def system_status():
    pct = volts = None
    if BATTERY_SCRIPT.exists():
        try:
            out = subprocess.run(["python3", str(BATTERY_SCRIPT)], capture_output=True, text=True, timeout=2).stdout
            m = re.search(r"(\d+(?:\.\d+)?)V", out)
            volts = float(m.group(1)) if m else None
            out = subprocess.run(["python3", str(BATTERY_SCRIPT), "--percentage_only"], capture_output=True, text=True, timeout=2).stdout
            pct = float(out.strip())
        except Exception:
            pass
    temps = psutil.sensors_temperatures()
    cpu_temp = temps["cpu_thermal"][0].current if "cpu_thermal" in temps else None
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=1).close()
        internet = "online"
    except OSError:
        internet = "offline"
    return {
        "battery": {"percentage": pct, "is_charging": False, "voltage": volts},
        "cpu": {"usage": psutil.cpu_percent(), "temperature": cpu_temp},
        "services": {"robot": "active" if tools else "inactive", "internet": internet},
    }


@app.post("/system/volume")
def system_volume(req: Volume):
    level = max(0.0, min(req.volume / 100.0, 1.5))
    r = subprocess.run(["wpctl", "set-volume", "@DEFAULT_SINK@", str(level)], capture_output=True, text=True)
    return {"success": r.returncode == 0, "message": f"volume set to {req.volume}%" if r.returncode == 0 else r.stderr}


@app.get("/health")
def health():
    return {"status": "running", "tools": tools is not None}


# The built web app (npm run build in web/ -> web/dist/) is served at /, so the robot's own
# browser, or a laptop through an SSH tunnel, opens everything from one address.
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")


class RealClock:
    """The pretend robot's clock, in real time (the desk tests use a fake one)."""

    @property
    def now(self) -> float:
        return time.monotonic()


def start_pretend_robot() -> None:
    """--sim: the tests' fake controller manager and follower, every leg controller off (lying down)."""
    import robot_api
    from tricks import load_tricks

    pretend.ROBOT = pretend.FakeRobot(RealClock())
    made = load_tricks()
    names = (set(robot_api.LEG_CONTROLLERS) | {t.controller for t in made.values() if t.kind == "policy"}
             | {t.walk_controller for t in made.values() if t.walks})
    pretend.ROBOT.controllers = {n: "inactive" for n in names}
    for name in ("joint_state_broadcaster", "imu_sensor_broadcaster"):
        pretend.ROBOT.controllers[name] = "active"
    log.info("pretend robot: %d leg controllers, nothing moves", len(names))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=port)
