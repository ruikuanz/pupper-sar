"""Pupper, as seen from Python: the one file in this project that talks to ROS.

Both components use the same few things from the robot, so they live here:

    pupper = Pupper()                  inside a running asyncio loop
    await pupper.stand_up()            -> Result: the walking policy takes the legs
    await pupper.lie_down()            -> Result: motors relaxed
    pupper.send_speed(vx, vy, wz_deg)  one speed command (+vx forward, +vy left, +wz counter-clockwise)
    pupper.play_reference(data)        a joint-angle table for the walking policy to follow once
                                       (skipped, with a warning, where it can't: walk_controller.py)
    pupper.camera_jpeg()               the newest camera picture (JPEG bytes), or None
    pupper.yaw_deg()                   heading from the IMU (degrees, unwrapped), or None
    pupper.problem()                   why it can't move right now ("" if it can)

The legs. One ros2_control controller holds the leg joints at a time. Standing
up is one controller switch: ours (sar_walk, see walk_controller.py) on, every
other leg controller off, in the same request, so there is never a moment with
two policies commanding the motors. Lying down is the same switch with nothing
turned on.

Speed commands go to /llm_cmd_vel, not /cmd_vel: a mux in between lets the
joystick outrank us and forgets us after half a second of silence (see
launch/pupper.launch.py). So the callers send a command 20 times a second for
as long as they want Pupper to move, and stopping is just not sending.

Falls. A leg controller whose body tilts past its limit latches an emergency
stop: it stays "active" but only damps the joints. Nothing on a topic says so;
the controller logs "Emergency stop triggered". We listen for that line on
/rosout, remember that Pupper fell, and refuse to move. Standing up again
first checks the IMU: level within SAFE_TILT_DEG and still for SAFE_STILL_S,
so it does not start a policy while Pupper is on its side or in someone's
hands. Then it turns the latched controller off and ours on, which clears the
latch.

The e-stop. The joystick's e-stop relaxes the platform's own controllers and
publishes /emergency_stop. Ours is not on its list, so we relax it ourselves.

ROS callbacks run on their own thread (the executor's). They only write plain
attributes or hand work to the asyncio loop; nothing here blocks the server.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass

import rclpy
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import Log
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Imu
from std_msgs.msg import Empty, Float32MultiArray

import walk_controller
from walk_controller import CONTROLLER_TYPE, LOADED, WALK_CONTROLLER, WALK_READY_S

log = logging.getLogger("pupper")

# The states: whatever holds the legs.
WALKING = "walking"   # sar_walk holds the legs (standing still is walking at speed 0)
IDLE = "idle"         # nothing holds the legs: Pupper lies down
FALLEN = "fallen"     # a tilt e-stop latched: the motors only damp until a safe stand-up

# The platform's own leg controllers (the joystick's). Anything else of the
# walking-policy type that turns up in the controller list is added to these.
PLATFORM_LEG_CONTROLLERS = {
    "neural_controller",
    "neural_controller_default",
    "neural_controller_default_alt",
    "neural_controller_three_legged",
    "neural_controller_student",
}

# Standing up again after a fall: the body must be this level and this still
# (every IMU sample over SAFE_STILL_S) before a controller is started.
SAFE_TILT_DEG = 20.0
SAFE_GYRO_RAD_S = 0.5
SAFE_STILL_S = 1.0
SAFE_WAIT_S = 4.0   # how long standing up waits for Pupper to settle
ESTOP_LOG = "Emergency stop triggered"


@dataclass
class Result:
    ok: bool
    message: str


def tilt_deg(qx: float, qy: float) -> float:
    """Angle between the body's up axis and the world's, from the orientation quaternion.

    Rotating the unit z axis by q gives a z component of 1 - 2(x^2 + y^2); the
    angle is its arccosine. Yaw (q.z, q.w) drops out, as it should.
    """
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))))


def yaw_deg(qx: float, qy: float, qz: float, qw: float) -> float:
    """Rotation about the up axis, in (-180, 180]."""
    return math.degrees(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


class Unwrap:
    """Turns a heading that jumps at +-180 into one that keeps counting.

    A dog that turns left twice is at +720, not back at 0: the navigator
    subtracts headings, and a jump of 360 would look like a spin.
    """

    def __init__(self) -> None:
        self.value: float | None = None
        self._turns = 0.0

    def update(self, angle_deg: float) -> float:
        if self.value is not None:
            last = self.value - self._turns
            if angle_deg - last > 180.0:
                self._turns -= 360.0
            elif angle_deg - last < -180.0:
                self._turns += 360.0
        self.value = angle_deg + self._turns
        return self.value


class Stillness:
    """A short memory of IMU samples, to answer: is Pupper upright and at rest?"""

    def __init__(self) -> None:
        self.samples: collections.deque = collections.deque()   # (time, tilt deg, gyro rad/s)

    def add(self, now: float, tilt: float, gyro: float) -> None:
        self.samples.append((now, tilt, gyro))
        while self.samples and self.samples[0][0] < now - SAFE_STILL_S - 0.5:
            self.samples.popleft()

    def upright(self, now: float) -> tuple[bool, str]:
        samples = list(self.samples)   # the ROS thread keeps appending
        recent = [s for s in samples if s[0] >= now - SAFE_STILL_S]
        if not recent or now - recent[-1][0] > 0.2:
            return False, "no IMU data, so it can't check that Pupper is upright"
        if now - samples[0][0] < SAFE_STILL_S:
            return False, "not enough IMU data yet"
        tilt = max(s[1] for s in recent)
        gyro = max(s[2] for s in recent)
        if tilt > SAFE_TILT_DEG:
            return False, f"Pupper is tilted {tilt:.0f} degrees (needs under {SAFE_TILT_DEG:.0f}); set it upright on its feet"
        if gyro > SAFE_GYRO_RAD_S:
            return False, "Pupper is still moving or being handled; let it rest on the ground"
        return True, f"upright ({tilt:.0f} degrees) and still"


def plan_switch(states: dict[str, str], leg_controllers: set[str], target: str | None) -> tuple[list[str], list[str]]:
    """The controller switch that leaves ``target`` (or nothing) holding the legs.

    -> (activate, deactivate): only what has to change, so a switch that is
    already done is two empty lists.
    """
    activate = [target] if target and states.get(target) != "active" else []
    deactivate = sorted(c for c in leg_controllers if c != target and states.get(c) == "active")
    return activate, deactivate


class Pupper:
    def __init__(self) -> None:
        if not rclpy.ok():
            rclpy.init()
        self.loop = asyncio.get_running_loop()   # ROS callbacks hand work to it
        self.node = Node("pupper_sar")
        self.switch_client = self.node.create_client(SwitchController, "/controller_manager/switch_controller")
        self.list_client = self.node.create_client(ListControllers, "/controller_manager/list_controllers")
        self.cmd_pub = self.node.create_publisher(Twist, "/llm_cmd_vel", 10)
        self.ref_pub = self.node.create_publisher(Float32MultiArray, f"/{WALK_CONTROLLER}/reference_trick", 10)

        self.state = IDLE
        self.fallen: str | None = None        # the controller that latched a tilt e-stop
        self.walk_since: float | None = None  # when sar_walk was last switched on
        self.estop_at = 0.0                   # when the joystick's e-stop was last pressed
        self.leg_controllers = {WALK_CONTROLLER} | PLATFORM_LEG_CONTROLLERS
        self.stillness = Stillness()
        self.heading = Unwrap()
        self.images: queue.Queue = queue.Queue(maxsize=1)
        self.references = walk_controller.references_supported()
        self._warned_references = False

        self.node.create_subscription(CompressedImage, "/camera/image_raw/compressed", self._on_image, 1)
        self.node.create_subscription(Imu, "/imu_sensor_broadcaster/imu", self._on_imu, 10)
        self.node.create_subscription(Log, "/rosout", self._on_rosout, 100)
        self.node.create_subscription(Empty, "/emergency_stop", self._on_estop, 10)

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()
        self.loop.create_task(self._adopt_running_state())

    # ---------------------------------------------------------------- what callers use
    async def stand_up(self) -> Result:
        """Hand the legs to the walking policy, and wait until it has faded in."""
        return await self._switch_to(WALK_CONTROLLER)

    async def lie_down(self) -> Result:
        self.send_speed(0, 0, 0)
        return await self._switch_to(None)

    def send_speed(self, vx: float, vy: float, wz_deg: float) -> None:
        """vx forward m/s, vy left m/s, wz_deg counter-clockwise deg/s. Forgotten after 0.5 s."""
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), math.radians(float(wz_deg))
        self.cmd_pub.publish(t)

    def play_reference(self, data: list[float]) -> None:
        """playback_s, active_s, cross_fade_s, then rows of 12 joint angles, flat (references/)."""
        if not self.references:
            self.references = walk_controller.references_supported()   # loaded since we started?
        if not self.references:
            if not self._warned_references:
                log.warning("this robot's walking controller cannot follow reference tables: jump and squat skipped "
                            "(see pupper/walk_controller.py)")
                self._warned_references = True
            return
        self.ref_pub.publish(Float32MultiArray(data=data))

    def camera_jpeg(self) -> bytes | None:
        try:
            return bytes(self.images.get_nowait().data)
        except queue.Empty:
            return None

    def yaw_deg(self) -> float | None:
        return self.heading.value

    def problem(self) -> str:
        if self.fallen:
            return "fell over: set Pupper upright, then press Stand up"
        if self.state != WALKING:
            return "not standing (lying down, or the e-stop was pressed)"
        return ""

    # ---------------------------------------------------------------- ROS callbacks (executor thread)
    def _on_image(self, msg) -> None:
        try:
            self.images.get_nowait()   # keep only the newest
        except queue.Empty:
            pass
        self.images.put_nowait(msg)

    def _on_imu(self, msg) -> None:
        q, w = msg.orientation, msg.angular_velocity
        self.stillness.add(time.monotonic(), tilt_deg(q.x, q.y), math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z))
        self.heading.update(yaw_deg(q.x, q.y, q.z, q.w))

    def _on_rosout(self, msg) -> None:
        controller = msg.name.rsplit(".", 1)[-1]
        if msg.msg != ESTOP_LOG or controller not in self.leg_controllers:
            return
        if time.monotonic() - self.estop_at < 2.0:
            return   # the joystick's e-stop, handled below
        self.loop.call_soon_threadsafe(self._on_fall, controller)

    def _on_estop(self, _msg) -> None:
        self.estop_at = time.monotonic()
        asyncio.run_coroutine_threadsafe(self._relax_after_estop(), self.loop)

    # ---------------------------------------------------------------- falls and the e-stop (asyncio loop)
    def _on_fall(self, controller: str) -> None:
        if self.fallen:
            return
        self.fallen = controller
        self.state = FALLEN
        self.send_speed(0, 0, 0)
        log.warning("tilt e-stop in %s: Pupper fell over; waiting for a safe stand-up", controller)

    async def _relax_after_estop(self) -> None:
        self.send_speed(0, 0, 0)
        self.fallen = None   # a person stopped it on purpose; standing up is their call
        self.state = IDLE
        self.walk_since = None
        states = await self._controller_states()
        _, active = plan_switch(states, self.leg_controllers, None)
        if active:
            await self._switch(deactivate=active, strict=False)
        log.warning("e-stop: relaxed %s", active or "nothing (no leg controller was active)")

    async def _adopt_running_state(self) -> None:
        """A restarted server finds the legs as the last one left them."""
        states = await self._controller_states()
        if states.get(WALK_CONTROLLER) == "active" and self.state == IDLE:
            self.state = WALKING
            log.info("found %s already holding the legs", WALK_CONTROLLER)

    # ---------------------------------------------------------------- the controller switch
    async def _call(self, client, request, timeout: float = 3.0):
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        return future.result() if future.done() else None

    async def _controller_states(self) -> dict[str, str]:
        res = await self._call(self.list_client, ListControllers.Request())
        if res is None:
            return {}
        # Any other walking-policy controller someone loaded holds the legs too.
        self.leg_controllers |= {c.name for c in res.controller if c.type == CONTROLLER_TYPE}
        return {c.name: c.state for c in res.controller}

    async def _switch(self, activate: list[str] = (), deactivate: list[str] = (), strict: bool = True):
        req = SwitchController.Request()
        req.activate_controllers = list(activate)
        req.deactivate_controllers = list(deactivate)
        req.strictness = SwitchController.Request.STRICT if strict else SwitchController.Request.BEST_EFFORT
        return await self._call(self.switch_client, req)

    async def _wait_upright(self) -> tuple[bool, str]:
        deadline = time.monotonic() + SAFE_WAIT_S
        while True:
            ok, why = self.stillness.upright(time.monotonic())
            if ok or time.monotonic() > deadline:
                return ok, why
            await asyncio.sleep(0.1)

    async def _wait_walk_ready(self) -> None:
        wait = (self.walk_since or time.monotonic()) + WALK_READY_S - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)

    async def _switch_to(self, target: str | None) -> Result:
        states = await self._controller_states()
        if not states:
            return Result(False, "controller_manager is not running; is the robot stack launched? (pupper/stack.sh start)")
        recovered = ""
        if self.fallen and target:
            ok, why = await self._wait_upright()
            if not ok:
                return Result(False, f"won't stand up: {why}")
            # Turn every latched controller off first: turning one on again clears its e-stop.
            _, latched = plan_switch(states, self.leg_controllers, None)
            if latched:
                await self._switch(deactivate=latched, strict=False)
                states = await self._controller_states()
            log.info("recovering from the fall in %s: %s", self.fallen, why)
            self.fallen, self.walk_since, self.state = None, None, IDLE
            recovered = "recovered from the fall and stood back up; "
        if target and states.get(target) not in LOADED:
            # The stack was started by something other than our launch file: load it now.
            log.warning("loading %s (it is %s)", target, states.get(target, "not loaded"))
            try:
                await asyncio.to_thread(walk_controller.load)
            except Exception as error:   # no platform package, no policy: say so instead of crashing the server
                return Result(False, f"could not load {target}: {error}")
            states = await self._controller_states()
            if states.get(target) not in LOADED:
                return Result(False, f"could not load {target}; see the robot log")
        if states.get(WALK_CONTROLLER) == "active" and self.walk_since is None:
            self.walk_since = time.monotonic() - WALK_READY_S   # it was walking before we looked
        activate, deactivate = plan_switch(states, self.leg_controllers, target)
        if activate or deactivate:
            res = await self._switch(activate, deactivate)
            if res is None or not res.ok:
                why = "no answer from controller_manager" if res is None else "refused by controller_manager"
                log.warning("switch %s -> %s: %s", deactivate, activate, why)
                return Result(False, f"controller switch failed: {why}")
            if WALK_CONTROLLER in activate:
                self.walk_since = time.monotonic()
            if WALK_CONTROLLER in deactivate:
                self.walk_since = None
        if self.fallen and target is None:
            return Result(True, "motors relaxed; Pupper is still on its side")   # stays FALLEN until a safe stand-up
        self.state = WALKING if target else IDLE
        if target:
            await self._wait_walk_ready()
        return Result(True, recovered + ("ok" if activate or deactivate else "already there"))
