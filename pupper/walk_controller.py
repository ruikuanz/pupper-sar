"""The walking controller this project drives: what it is, and how it gets loaded.

Pupper's legs are held by one ros2_control controller at a time. The platform
(the pupperv3 ROS workspace every Pupper v3 ships with) has a controller type,
neural_controller/NeuralController, that runs a trained walking policy 50 times
a second, and a ready-made instance of it for the joystick. We load a second
instance of the same policy under our own name, sar_walk, with three settings
changed:

    reference_tricks   on: the policy accepts a table of joint angles on
                       /sar_walk/reference_trick and follows it once, keeping
                       its balance. That is how Pupper jumps and squats.
    game_buttons       off: the joystick's trick buttons stay out of our way.
    speed caps         1.5 m/s instead of the joystick instance's walking pace
                       (we command far less; see moves.py and navigator.py).

Everything else (gains, the policy file, the tilt e-stop angle) is copied from
the platform's own instance, read at launch time from the installed package, so
this follows whatever policy the robot has.

Two kinds of robot. A Pupper set up with the course's walking-policy controller
(Stanford CS 123's pupper_gait_deploy, installed into the platform workspace)
has an instance called neural_controller_default whose controller supports
reference tricks: everything here works. A stock Pupper has only
neural_controller, which walks but cannot follow a reference table. There we
copy that one and leave the three settings above alone: walking, running in
place, sidestepping and the whole echolocation component work; the game's jump
and squat are skipped, with a warning in the log (references_supported()).

The assistant (gemini/) also plays tricks, each a trained policy on its own
controller of the same type (trick_<name>, and walk_<name> for a trick that
walks in its own stance). Their parameters go in the same file, built from the
same platform instance (gemini/tricks.py has the settings). They are loaded at
launch only for the assistant (`--tricks`), and otherwise on first use. Trick
policies need the CS 123 controller too; on a stock robot none are written.

Used four ways:

    launch/pupper.launch.py     runs `python3 walk_controller.py [--tricks]` once
                                the platform's own controllers are up
    pupper_ros.py               calls load() if sar_walk is missing when asked
                                to stand (the stack was started by something else)
    gemini/robot_api.py         calls load([...]) for a trick controller that is missing
    stack.sh                    waits for READY_FILE

No ROS imports at the top, so the tests can import it on a laptop.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

WALK_CONTROLLER = "sar_walk"
CONTROLLER_TYPE = "neural_controller/NeuralController"
PLATFORM_PACKAGE = "neural_controller"            # the platform package with the policy and its settings
# The instance whose settings we start from: the first of these the platform has.
# The first one comes with reference-trick support; the second is stock.
PLATFORM_CONTROLLERS = ("neural_controller_default", "neural_controller")
WALK_SPEED_CAP = 1.5   # m/s
# The controller holds the default pose for init_duration, then fades the policy
# in over fade_in_duration (2 s + 2 s). Speed commands and reference tricks sent
# before that land on a half-started controller.
WALK_READY_S = 4.2

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = Path("/tmp/pupper-sar")
PARAM_FILE = RUN_DIR / "walk_controller.yaml"
READY_FILE = RUN_DIR / "controllers_ready"   # start.sh waits for this
RETRIES = 2
LOADED = {"inactive", "active"}  # loaded and configured: ready to switch to


def walk_params(base: dict, references: bool = True) -> dict:
    """sar_walk's parameters, from the platform instance's (``base``).

    references: the platform's controller supports reference tricks (and the
    other settings that come with them). Without it only the type is set.
    """
    params = dict(base)
    params.pop("chord_partner", None)   # a joystick pairing that belongs to the platform instance
    params["type"] = CONTROLLER_TYPE
    if references:
        params.update(
            game_buttons=False,
            speed_cap=WALK_SPEED_CAP,
            max_vx_forward=WALK_SPEED_CAP,
            max_vx_backward=WALK_SPEED_CAP,
            reference_tricks=True,
        )
    return params


def choose_platform_controller(blocks: dict) -> tuple[str, bool]:
    """The platform config's blocks -> (the instance to copy, does it support reference tricks)."""
    for name in PLATFORM_CONTROLLERS:
        if name in blocks:
            return name, name == PLATFORM_CONTROLLERS[0]
    raise KeyError(f"no walking-policy controller ({' or '.join(PLATFORM_CONTROLLERS)}) in the platform config")


def references_supported(path: Path = PARAM_FILE) -> bool:
    """Does the sar_walk this robot loaded follow reference tables (the jump, the squat)?"""
    try:
        return "reference_tricks: true" in path.read_text()
    except OSError:
        return False


def expand_package_paths(text: str, share_directory) -> str:
    """Replace every $(find-pkg-share <pkg>) in ``text`` with that package's path.

    Launch files do this substitution for the platform's config; the controller
    spawner reads our parameter file as plain YAML, so we do it ourselves.
    """
    return re.sub(r"\$\(find-pkg-share\s+([^)\s]+)\s*\)", lambda m: str(share_directory(m.group(1))), text)


def trick_params(base: dict) -> dict[str, dict]:
    """{controller: parameters} for every policy trick in gemini/tricks (and walking tricks' walking policies)."""
    sys.path.insert(0, str(ROOT / "gemini"))
    import tricks  # noqa: E402  (gemini/tricks.py: no ROS imports)

    params = {}
    for t in tricks.load_tricks().values():
        if t.kind != "policy":
            continue
        params[t.controller] = {"type": CONTROLLER_TYPE, **tricks.controller_params(t, base)}
        if t.walks:
            params[t.walk_controller] = {"type": CONTROLLER_TYPE, **tricks.walk_policy_params(t, base)}
    return params


def trick_controller_names() -> list[str]:
    """The trick controllers write_param_file wrote (none on a stock robot)."""
    import yaml

    try:
        return [n for n in yaml.safe_load(PARAM_FILE.read_text()) if n != WALK_CONTROLLER]
    except (OSError, TypeError):
        return []


def write_param_file(path: Path = PARAM_FILE) -> Path:
    """Write the spawner's parameter file: sar_walk, and every trick controller. Needs the ROS environment."""
    import yaml
    from ament_index_python.packages import get_package_share_directory

    config = Path(get_package_share_directory(PLATFORM_PACKAGE)) / "launch" / "config.yaml"
    blocks = yaml.safe_load(config.read_text())
    source, references = choose_platform_controller(blocks)
    params = walk_params(blocks[source]["ros__parameters"], references)
    if not references:
        print(f"walk_controller: {config} has only {source}: walking works, the jump and squat will be skipped", flush=True)
    controllers = {WALK_CONTROLLER: params}
    if references:   # trick policies need the controller that supports them
        controllers.update(trick_params(blocks[source]["ros__parameters"]))
    for p in controllers.values():
        if isinstance(p.get("model_path"), str):
            p["model_path"] = expand_package_paths(p["model_path"], get_package_share_directory)

    class NoAliases(yaml.SafeDumper):   # ROS parameter files reject YAML aliases
        def ignore_aliases(self, data):
            return True

    path.parent.mkdir(parents=True, exist_ok=True)
    blocks_out = {name: {"ros__parameters": p} for name, p in controllers.items()}
    path.write_text(yaml.dump(blocks_out, Dumper=NoAliases, sort_keys=False))
    return path


def parse_controller_list(text: str) -> dict[str, str]:
    """`ros2 control list_controllers` output -> {name: state}.

    Each line is "<name>  <type>  <state>", the state possibly in ANSI colours.
    """
    states = {}
    for line in re.sub(r"\x1b\[[0-9;]*m", "", text).splitlines():
        words = line.split()
        if len(words) >= 3:
            states[words[0]] = words[2]
    return states


def controller_states() -> dict[str, str]:
    try:
        out = subprocess.run(["ros2", "control", "list_controllers"], capture_output=True, text=True, timeout=20).stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}
    return parse_controller_list(out)


def spawn(names=(WALK_CONTROLLER,), param_file: Path = PARAM_FILE) -> bool:
    """Load and configure controllers, inactive. True if the spawner reported success.

    One spawner gives up on the rest of its list at the first controller that
    fails (for example a policy file that took longer to parse than a service
    call waited), so main() checks each one afterwards.
    """
    cmd = [
        "ros2", "run", "controller_manager", "spawner", *names,
        "--controller-manager", "/controller_manager",
        "--controller-manager-timeout", "60", "--service-call-timeout", "60",
        "--inactive", "--param-file", str(param_file),
    ]
    return subprocess.run(cmd).returncode == 0


def load(names=(WALK_CONTROLLER,)) -> bool:
    """Write the parameter file and load these controllers now (a few seconds each; blocks)."""
    return spawn(list(names), write_param_file())


def wait_for_platform_controllers(timeout: float = 90.0) -> bool:
    """The policy file is large and the controller manager is busy while it parses
    one, so loading ours at the same time as the platform's makes spawners time
    out. Go after them."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        states = controller_states()
        if states.get("imu_sensor_broadcaster") == "active" and any(c in states for c in PLATFORM_CONTROLLERS):
            return True
        time.sleep(1.0)
    return False


def main(argv: list[str]) -> int:
    """python3 walk_controller.py [--tricks]: load sar_walk (and every trick controller), then write READY_FILE."""
    READY_FILE.unlink(missing_ok=True)
    if not wait_for_platform_controllers():
        print("walk_controller: the platform's controllers did not come up; loading ours anyway", flush=True)
    param_file = write_param_file()
    names = [WALK_CONTROLLER] + (trick_controller_names() if "--tricks" in argv else [])
    spawn(names, param_file)
    for attempt in range(RETRIES):
        states = controller_states()
        missing = [n for n in names if states.get(n) not in LOADED]
        if not missing:
            break
        print(f"walk_controller: not ready after attempt {attempt + 1}: "
              + ", ".join(f"{n} ({states.get(n, 'not loaded')})" for n in missing), flush=True)
        for n in missing:
            spawn([n], param_file)
    states = controller_states()
    failed = [n for n in names if states.get(n) not in LOADED]
    summary = f"{len(names) - len(failed)}/{len(names)} controllers ready"
    if failed:
        summary += "; FAILED: " + ", ".join(f"{n} ({states.get(n, 'not loaded')})" for n in failed)
    if not references_supported(param_file):
        summary += " (walking only: no reference or trick policies on this robot)"
    print("walk_controller: " + summary, flush=True)
    READY_FILE.parent.mkdir(parents=True, exist_ok=True)
    READY_FILE.write_text(summary + "\n")
    return 0 if WALK_CONTROLLER not in failed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
