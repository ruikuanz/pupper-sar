"""Desk-side tests for the robot layer (pupper/): no robot, no ROS.

    python3 tests/test_pupper.py            # everything
    python3 tests/test_pupper.py legs       # one part: math, controller, legs

pupper_ros.py is the one file that imports ROS, and a laptop has no ROS. So
this file builds a pretend one first: message classes that are plain objects,
a node that remembers what was published, and a controller manager that keeps
a table of controllers and applies switch requests to it. Then it runs the
real pupper_ros.Pupper against it and checks the things that matter on the
floor: which controllers a stand-up turns on and off, what a fall does, what
the e-stop does.

What this cannot check is ROS itself: topic names, message layouts and the
launch file are only as right as the robot says they are.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pupper"))

import walk_controller as W  # noqa: E402

FAILED = []
COUNT = [0]


def check(name: str, ok: bool, got=None, want=None) -> None:
    COUNT[0] += 1
    if ok:
        print(f"  ok    {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}\n          got:    {got}\n          wanted: {want}")


# ------------------------------------------------------------------ a pretend ROS
class Message:
    """Any ROS message: an object you can hang fields on."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class Twist(Message):
    def __init__(self):
        super().__init__(linear=Message(x=0.0, y=0.0, z=0.0), angular=Message(x=0.0, y=0.0, z=0.0))


class SwitchRequest(Message):
    STRICT, BEST_EFFORT = 2, 1

    def __init__(self):
        super().__init__(activate_controllers=[], deactivate_controllers=[], strictness=0)


class Service:
    Request = Message


class SwitchController:
    Request = SwitchRequest


class Done:
    """A service call that has already been answered."""

    def __init__(self, result):
        self._result = result

    def done(self) -> bool:
        return True

    def result(self):
        return self._result


class ControllerManager:
    """The table of controllers, and what a switch request does to it."""

    def __init__(self, controllers: dict[str, tuple[str, str]]):
        self.controllers = dict(controllers)   # name -> (type, state)
        self.running = True
        self.switches: list[tuple[list[str], list[str], int]] = []

    def state(self, name: str) -> str:
        return self.controllers[name][1]

    def list(self):
        return Message(controller=[Message(name=n, type=t, state=s) for n, (t, s) in self.controllers.items()])

    def switch(self, request):
        self.switches.append((list(request.activate_controllers), list(request.deactivate_controllers), request.strictness))
        names = [*request.activate_controllers, *request.deactivate_controllers]
        if request.strictness == SwitchRequest.STRICT and any(n not in self.controllers for n in names):
            return Message(ok=False)
        for name in request.deactivate_controllers:
            if name in self.controllers:
                self.controllers[name] = (self.controllers[name][0], "inactive")
        for name in request.activate_controllers:
            if name in self.controllers:
                self.controllers[name] = (self.controllers[name][0], "active")
        return Message(ok=True)


MANAGER: ControllerManager | None = None


class Client:
    def __init__(self, name: str):
        self.name = name

    def wait_for_service(self, timeout_sec: float = 0.0) -> bool:
        return MANAGER.running

    def call_async(self, request):
        return Done(MANAGER.switch(request) if self.name.endswith("switch_controller") else MANAGER.list())


class Publisher:
    def __init__(self):
        self.sent = []

    def publish(self, message) -> None:
        self.sent.append(message)


class Node:
    def __init__(self, name: str):
        self.name = name
        self.publishers: dict[str, Publisher] = {}
        self.callbacks: dict[str, object] = {}

    def create_client(self, _type, name: str) -> Client:
        return Client(name)

    def create_publisher(self, _type, topic: str, _depth) -> Publisher:
        return self.publishers.setdefault(topic, Publisher())

    def create_subscription(self, _type, topic: str, callback, _depth) -> None:
        self.callbacks[topic] = callback


class Executor:
    def add_node(self, node) -> None:
        pass

    def spin(self) -> None:
        pass


REFERENCES = {"ok": True}
W.references_supported = lambda path=None: REFERENCES["ok"]


def install_pretend_ros() -> None:
    def module(name: str, **contents):
        mod = types.ModuleType(name)
        mod.__dict__.update(contents)
        sys.modules[name] = mod
        return mod

    module("rclpy", ok=lambda: True, init=lambda: None)
    module("rclpy.executors", SingleThreadedExecutor=Executor)
    module("rclpy.node", Node=Node)
    module("controller_manager_msgs")
    module("controller_manager_msgs.srv", ListControllers=Service, SwitchController=SwitchController)
    module("geometry_msgs")
    module("geometry_msgs.msg", Twist=Twist)
    module("rcl_interfaces")
    module("rcl_interfaces.msg", Log=Message)
    module("sensor_msgs")
    module("sensor_msgs.msg", CompressedImage=Message, Imu=Message)
    module("std_msgs")
    module("std_msgs.msg", Empty=Message, Float32MultiArray=Message)


install_pretend_ros()
import pupper_ros as R  # noqa: E402

POLICY = W.CONTROLLER_TYPE
OTHER = "joint_state_broadcaster/JointStateBroadcaster"


def a_robot(**changes) -> ControllerManager:
    """A robot whose stack is up and whose legs nothing holds."""
    controllers = {
        "joint_state_broadcaster": (OTHER, "active"),
        "imu_sensor_broadcaster": ("imu_sensor_broadcaster/IMUSensorBroadcaster", "active"),
        "neural_controller_default": (POLICY, "inactive"),
        W.WALK_CONTROLLER: (POLICY, "inactive"),
    }
    for name, state in changes.items():
        if state is None:
            controllers.pop(name, None)
        else:
            controllers[name] = (controllers.get(name, (POLICY, ""))[0], state)
    return ControllerManager(controllers)


def feel(pupper, tilt_deg: float = 0.0, gyro: float = 0.0, seconds: float = 1.3) -> None:
    """Fill the IMU memory as if Pupper had been like this for ``seconds``, until now."""
    pupper.stillness.samples.clear()
    now = time.monotonic()
    for i in range(int(seconds / 0.02), -1, -1):
        pupper.stillness.add(now - i * 0.02, tilt_deg, gyro)


def log_line(controller: str, text: str = R.ESTOP_LOG) -> Message:
    return Message(name=f"controller_manager.{controller}", msg=text)


async def settle() -> None:
    """Let the work that ROS callbacks handed to the asyncio loop run."""
    for _ in range(5):
        await asyncio.sleep(0.01)


# ------------------------------------------------------------------ math
def test_math() -> None:
    print("orientation, heading, stillness")
    half = math.radians(30) / 2
    check("level: no tilt", abs(R.tilt_deg(0.0, 0.0)) < 1e-9, R.tilt_deg(0.0, 0.0), 0.0)
    check("rolled 30 degrees: tilt 30", abs(R.tilt_deg(math.sin(half), 0.0) - 30) < 1e-6, R.tilt_deg(math.sin(half), 0.0), 30)
    check("pitched 30 degrees: tilt 30", abs(R.tilt_deg(0.0, math.sin(half)) - 30) < 1e-6, R.tilt_deg(0.0, math.sin(half)), 30)
    check("turning on the spot is not tilt, and reads as yaw",
          abs(R.yaw_deg(0.0, 0.0, math.sin(half), math.cos(half)) - 30) < 1e-6, R.yaw_deg(0.0, 0.0, math.sin(half), math.cos(half)), 30)
    check("on its back: tilt 180", abs(R.tilt_deg(1.0, 0.0) - 180) < 1e-6, R.tilt_deg(1.0, 0.0), 180)

    heading = R.Unwrap()
    seen = [heading.update(a) for a in (170, 179, -179, -170, -90, 0, 90, 179, -179)]
    check("a heading that passes 180 keeps counting", seen == [170, 179, 181, 190, 270, 360, 450, 539, 541], seen, "170 ... 541")
    heading = R.Unwrap()
    seen = [heading.update(a) for a in (-170, -179, 179, 170)]
    check("... and the other way round", seen == [-170, -179, -181, -190], seen, [-170, -179, -181, -190])

    def memory(tilt, gyro, seconds=1.3, age=0.0):
        s = R.Stillness()
        for i in range(int(seconds / 0.02), -1, -1):
            s.add(100.0 - age - i * 0.02, tilt, gyro)
        return s.upright(100.0)

    check("level and at rest: upright", memory(3.0, 0.05)[0], memory(3.0, 0.05), True)
    check("on its side: not upright", not memory(80.0, 0.0)[0] and "tilted" in memory(80.0, 0.0)[1], memory(80.0, 0.0), "tilted")
    check("being carried: not upright", not memory(5.0, 1.5)[0] and "moving" in memory(5.0, 1.5)[1], memory(5.0, 1.5), "moving")
    check("only half a second of data: wait", not memory(0.0, 0.0, seconds=0.5)[0], memory(0.0, 0.0, seconds=0.5), "not enough")
    check("the IMU went quiet: not upright", not memory(0.0, 0.0, age=0.5)[0], memory(0.0, 0.0, age=0.5), "no IMU data")
    s = R.Stillness()
    for i in range(60, -1, -1):   # a bump 1.2 s ago, calm since: the bump is outside the window
        s.add(100.0 - i * 0.02, 0.0, 3.0 if i == 60 else 0.0)
    check("a bump more than a second ago is forgotten", s.upright(100.0)[0], s.upright(100.0), True)


# ------------------------------------------------------------------ the walking controller
def test_controller() -> None:
    print("the walking controller's settings")
    base = {"model_path": "$(find-pkg-share neural_controller)/launch/policy.json", "kps": [5.0] * 12,
            "chord_partner": "neural_controller_default_alt", "game_buttons": True, "speed_cap": 0.75,
            "max_body_angle": 0.9, "type": "neural_controller/NeuralController"}
    params = W.walk_params(base)
    check("reference tricks on, joystick tricks off", params["reference_tricks"] is True and params["game_buttons"] is False, params, "")
    check("the policy, its gains and its tilt limit are the platform's",
          all(params[k] == base[k] for k in ("model_path", "kps", "max_body_angle")), params, base)
    check("the joystick pairing is left behind", "chord_partner" not in params, list(params), "no chord_partner")
    check("the platform's settings are not touched", base["game_buttons"] is True and "chord_partner" in base, base, "")
    stock = W.walk_params({"model_path": "p.json", "kps": [5.0] * 12}, references=False)
    check("a stock robot: the policy is copied and nothing it doesn't know is added",
          stock == {"model_path": "p.json", "kps": [5.0] * 12, "type": W.CONTROLLER_TYPE}, stock, "")
    chosen = (W.choose_platform_controller({"neural_controller": {}, "neural_controller_default": {}}),
              W.choose_platform_controller({"neural_controller": {}, "joint_state_broadcaster": {}}))
    check("the course's controller is preferred; the stock one is the fallback, without reference tricks",
          chosen == (("neural_controller_default", True), ("neural_controller", False)), chosen, "")
    try:
        W.choose_platform_controller({"joint_state_broadcaster": {}})
        check("no walking policy at all: a clear error", False, "no error", "KeyError")
    except KeyError:
        check("no walking policy at all: a clear error", True)
    tricks = W.trick_params(base)
    sit = tricks.get("trick_sit", {})
    check("every policy trick in gemini/tricks gets a controller of the same type",
          len(tricks) >= 10 and all(p["type"] == W.CONTROLLER_TYPE for p in tricks.values()), sorted(tricks), "trick_*")
    check("... running its own policy file, with a short blend-in and no walking",
          sit.get("model_path", "").endswith("gemini/tricks/sit/policy.json") and Path(sit["model_path"]).exists()
          and sit.get("init_duration") == 0.5 and sit.get("max_vx_forward") == 0.0 and "chord_partner" not in sit, sit, "")
    path = W.expand_package_paths(base["model_path"], lambda pkg: f"/ws/share/{pkg}")
    check("$(find-pkg-share ...) becomes a path", path == "/ws/share/neural_controller/launch/policy.json", path, "")
    listing = ("joint_state_broadcaster joint_state_broadcaster/JointStateBroadcaster  \x1b[32mactive\x1b[0m\n"
               "sar_walk                neural_controller/NeuralController            \x1b[36minactive\x1b[0m\n")
    states = W.parse_controller_list(listing)
    check("the controller list is read through its colours", states == {"joint_state_broadcaster": "active", "sar_walk": "inactive"}, states, "")

    for name in ("jump.json",):
        ref = json.loads((ROOT / "pupper" / "references" / name).read_text())
        rows = ref["table"]
        check(f"{name}: rows of 12 joint angles, played in under 2 s",
              all(len(r) == 12 for r in rows) and 0 < ref["active_s"] <= ref["playback_s"] < 2.0, (len(rows), ref["playback_s"]), "")
        # Left and right legs mirror each other in a jump straight up: (hip, thigh, knee) pairs with opposite signs.
        mirrored = all(abs(r[i] + r[i + 3]) < 1e-6 for r in rows for i in (0, 1, 2, 6, 7, 8))
        check(f"{name}: left and right legs mirror each other", mirrored, mirrored, True)

    plan = R.plan_switch({"sar_walk": "inactive", "neural_controller_default": "active", "joint_state_broadcaster": "active"},
                         {"sar_walk", "neural_controller_default"}, "sar_walk")
    check("stand up: ours on, the other leg controller off, sensors left alone", plan == (["sar_walk"], ["neural_controller_default"]), plan, "")
    plan = R.plan_switch({"sar_walk": "active"}, {"sar_walk", "neural_controller_default"}, "sar_walk")
    check("already standing: nothing to switch", plan == ([], []), plan, ([], []))
    plan = R.plan_switch({"sar_walk": "active"}, {"sar_walk"}, None)
    check("lie down: ours off, nothing on", plan == ([], ["sar_walk"]), plan, ([], ["sar_walk"]))


# ------------------------------------------------------------------ the legs
async def legs() -> None:
    global MANAGER
    R.WALK_READY_S = 0.15   # the real fade-in is 4.2 s
    R.SAFE_WAIT_S = 0.1

    print("standing up and lying down")
    MANAGER = a_robot(neural_controller_default="active")
    pupper = R.Pupper()
    await settle()
    check("nothing holds the legs yet: cannot move", "not standing" in pupper.problem(), pupper.problem(), "not standing")
    began = time.monotonic()
    result = await pupper.stand_up()
    took = time.monotonic() - began
    check("stand up succeeds", result.ok, result, "ok")
    check("one strict switch: ours on and the joystick's off together",
          MANAGER.switches == [([W.WALK_CONTROLLER], ["neural_controller_default"], SwitchRequest.STRICT)], MANAGER.switches, "")
    check("it waits for the policy to fade in before saying so", took >= 0.14, round(took, 3), ">= 0.15")
    check("standing: free to move", pupper.problem() == "", pupper.problem(), "")
    result = await pupper.stand_up()
    check("standing up twice switches nothing", result.ok and len(MANAGER.switches) == 1, (result, MANAGER.switches), "")

    pupper.send_speed(0.3, -0.2, 90.0)
    twist = pupper.node.publishers["/llm_cmd_vel"].sent[-1]
    check("speed goes to the mux's topic, the turn in radians",
          (twist.linear.x, twist.linear.y) == (0.3, -0.2) and abs(twist.angular.z - math.pi / 2) < 1e-9,
          (twist.linear.x, twist.linear.y, twist.angular.z), (0.3, -0.2, 1.5708))
    pupper.play_reference([0.9, 0.8, 0.06] + [0.0] * 12)
    sent = pupper.node.publishers[f"/{W.WALK_CONTROLLER}/reference_trick"].sent
    check("a reference table goes to our controller", len(sent) == 1 and len(sent[0].data) == 15, len(sent), 1)

    REFERENCES["ok"] = False
    plain = R.Pupper()
    plain.play_reference([0.9, 0.8, 0.06] + [0.0] * 12)
    plain.play_reference([0.9, 0.8, 0.06] + [0.0] * 12)
    check("a controller without reference tricks: the table is not sent (one warning in the log)",
          f"/{W.WALK_CONTROLLER}/reference_trick" not in plain.node.publishers or
          not plain.node.publishers[f"/{W.WALK_CONTROLLER}/reference_trick"].sent, "", "")
    REFERENCES["ok"] = True

    result = await pupper.lie_down()
    check("lie down: ours off", result.ok and MANAGER.state(W.WALK_CONTROLLER) == "inactive", MANAGER.controllers, "")
    twist = pupper.node.publishers["/llm_cmd_vel"].sent[-1]
    check("... after a zero speed", (twist.linear.x, twist.linear.y, twist.angular.z) == (0.0, 0.0, 0.0), twist.linear.x, 0.0)
    check("lying down: cannot move", "not standing" in pupper.problem(), pupper.problem(), "not standing")

    print("a stack someone else started")
    MANAGER = a_robot(trick_spin="active", **{W.WALK_CONTROLLER: None})
    pupper = R.Pupper()
    loaded = []

    def pretend_load() -> bool:
        loaded.append(True)
        MANAGER.controllers[W.WALK_CONTROLLER] = (POLICY, "inactive")
        return True

    R.walk_controller.load = pretend_load
    result = await pupper.stand_up()
    check("our controller is missing: it is loaded first", result.ok and loaded == [True], (result, loaded), "")
    check("a policy we have never heard of is holding the legs: it is turned off too",
          MANAGER.state("trick_spin") == "inactive" and MANAGER.state(W.WALK_CONTROLLER) == "active", MANAGER.controllers, "")
    check("the sensors' controllers are never touched",
          MANAGER.state("joint_state_broadcaster") == "active" and MANAGER.state("imu_sensor_broadcaster") == "active", MANAGER.controllers, "")

    def failing_load() -> bool:
        raise KeyError("no neural_controller_default")

    MANAGER = a_robot(**{W.WALK_CONTROLLER: None})
    R.walk_controller.load = failing_load
    pupper = R.Pupper()
    result = await pupper.stand_up()
    check("it cannot be loaded: a clear no, and nothing switched", not result.ok and "could not load" in result.message and not MANAGER.switches, result, "")

    MANAGER = a_robot()
    MANAGER.running = False
    pupper = R.Pupper()
    result = await pupper.stand_up()
    check("no robot stack: a clear no", not result.ok and "stack" in result.message, result, "")

    print("a server that restarts while Pupper stands")
    MANAGER = a_robot(**{W.WALK_CONTROLLER: "active"})
    pupper = R.Pupper()
    await settle()
    check("it finds Pupper standing and carries on", pupper.problem() == "", pupper.problem(), "")
    began = time.monotonic()
    result = await pupper.stand_up()
    check("... without waiting for a fade-in that is long over", result.ok and time.monotonic() - began < 0.1, round(time.monotonic() - began, 3), "< 0.1")

    print("falling over")
    MANAGER = a_robot()
    pupper = R.Pupper()
    await pupper.stand_up()
    rosout = pupper.node.callbacks["/rosout"]
    rosout(log_line(W.WALK_CONTROLLER, "Policy output clipped"))
    rosout(log_line("joint_state_broadcaster"))
    await settle()
    check("other log lines, and other nodes' e-stops, are not a fall", pupper.problem() == "", pupper.problem(), "")
    speeds = len(pupper.node.publishers["/llm_cmd_vel"].sent)
    rosout(log_line(W.WALK_CONTROLLER))
    await settle()
    check("our controller logs its tilt e-stop: Pupper fell", "fell over" in pupper.problem(), pupper.problem(), "fell over")
    sent = pupper.node.publishers["/llm_cmd_vel"].sent
    check("... and is told to stand still at once", len(sent) == speeds + 1 and sent[-1].linear.x == 0.0, len(sent) - speeds, 1)

    feel(pupper, tilt_deg=75.0)
    switches = len(MANAGER.switches)
    result = await pupper.stand_up()
    check("on its side: it won't stand up", not result.ok and "tilted" in result.message, result, "tilted")
    check("... and no controller is touched", len(MANAGER.switches) == switches, MANAGER.switches[switches:], [])
    feel(pupper, tilt_deg=2.0, gyro=2.0)
    result = await pupper.stand_up()
    check("upright but in someone's hands: it won't stand up", not result.ok and "moving" in result.message, result, "moving")

    result = await pupper.lie_down()
    check("lie down still works after a fall: the latched controller is relaxed",
          result.ok and MANAGER.state(W.WALK_CONTROLLER) == "inactive", (result, MANAGER.controllers), "")
    check("... but Pupper is still known to have fallen", "fell over" in pupper.problem(), pupper.problem(), "fell over")
    MANAGER.controllers[W.WALK_CONTROLLER] = (POLICY, "active")   # as it is right after a real fall: latched, still "active"

    feel(pupper, tilt_deg=2.0, gyro=0.02)
    switches = len(MANAGER.switches)
    result = await pupper.stand_up()
    check("set upright and left alone: it stands up", result.ok and "recovered" in result.message, result, "recovered")
    off_then_on = [s[:2] for s in MANAGER.switches[switches:]]
    check("the latched controller is turned off, then on again (that clears the latch)",
          off_then_on == [([], [W.WALK_CONTROLLER]), ([W.WALK_CONTROLLER], [])], off_then_on, "")
    check("standing again: free to move", pupper.problem() == "", pupper.problem(), "")

    print("the e-stop")
    pupper.node.callbacks["/emergency_stop"](Message())
    rosout(log_line(W.WALK_CONTROLLER))   # the controller logs the same line for the joystick's e-stop
    await settle()
    check("the e-stop relaxes our controller too", MANAGER.state(W.WALK_CONTROLLER) == "inactive", MANAGER.controllers, "")
    check("the last switch was best-effort (relax whatever can be relaxed)", MANAGER.switches[-1][2] == SwitchRequest.BEST_EFFORT, MANAGER.switches[-1], "")
    check("an e-stop is not a fall: Pupper is lying down, by a person's choice",
          "not standing" in pupper.problem(), pupper.problem(), "not standing")
    result = await pupper.stand_up()
    check("... and may stand up again without the upright check", result.ok, result, "ok")

    print("the camera, the heading")
    check("no picture yet", pupper.camera_jpeg() is None, pupper.camera_jpeg(), None)
    image = pupper.node.callbacks["/camera/image_raw/compressed"]
    image(Message(data=b"first"))
    image(Message(data=b"second"))
    check("only the newest picture is kept", pupper.camera_jpeg() == b"second" and pupper.camera_jpeg() is None, "", "")
    check("no heading before the IMU speaks", pupper.yaw_deg() is None, pupper.yaw_deg(), None)
    imu = pupper.node.callbacks["/imu_sensor_broadcaster/imu"]
    for angle in (170, 179, -179, -170):
        half = math.radians(angle) / 2
        imu(Message(orientation=Message(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half)), angular_velocity=Message(x=0.0, y=0.0, z=0.5)))
    check("the heading turns through 180 without jumping", abs(pupper.yaw_deg() - 190) < 1e-6, pupper.yaw_deg(), 190)


def test_legs() -> None:
    asyncio.run(legs())


TESTS = {"math": [test_math], "controller": [test_controller], "legs": [test_legs]}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(TESTS)
    for name in wanted:
        if name not in TESTS:
            sys.exit(f"unknown part {name!r}; choose from {', '.join(TESTS)}")
        for test in TESTS[name]:
            test()
    print(f"\n{COUNT[0] - len(FAILED)} of {COUNT[0]} checks passed" + (f"; FAILED: {len(FAILED)}" if FAILED else ""))
    sys.exit(1 if FAILED else 0)
