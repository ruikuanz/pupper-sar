"""The robot stack: everything that has to run on Pupper before our servers start.

    ros2 launch pupper/launch/pupper.launch.py        (pupper/stack.sh does this)

It starts the platform's own launch (motor drivers, IMU, camera, the walking
policy, the joystick and its e-stop) and makes sure of three things:

  - cmd_vel_mux, between every source of speed commands and the walking policy.
    The joystick publishes /teleop_cmd_vel, we publish /llm_cmd_vel, the mux
    forwards one of them to /cmd_vel. The joystick outranks us, and a source
    that goes quiet for half a second is dropped: if our server dies, Pupper
    stops walking. The stock platform launch already has the mux; the course's
    variant of it does not, and then it is added here, with the joystick
    re-routed through it.
  - a fixed white balance on the camera (camera_params.yaml). Auto white
    balance leaves this camera pink indoors, which matters when the vision is
    a hue test.
  - sar_walk, our instance of the walking policy (walk_controller.py).

For the assistant (SAR_ASSISTANT=1, which `stack.sh start --assistant` sets) it
also loads every trick controller, and makes sure the platform's person
follower (the Hailo person detector and the follower node) is running: the
stock launch starts them, the course's variant does not.

Every package named here is in the pupperv3 workspace the robot ships with;
nothing else needs to be cloned.
"""
import os
import sys
from pathlib import Path

import yaml

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

HERE = Path(__file__).resolve().parent
PARAMS = str(HERE / "pupper.yaml")
sys.path.insert(0, str(HERE.parent))
from walk_controller import PLATFORM_PACKAGE  # noqa: E402

PLATFORM_CONFIG = PathJoinSubstitution([FindPackageShare(PLATFORM_PACKAGE), "launch", "config.yaml"])


def camera_white_balance():
    """Set the gains once the platform's camera node is up (it starts with its own parameters)."""
    wb = yaml.safe_load((HERE.parent / "camera_params.yaml").read_text()).get("white_balance", {})
    if wb.get("awb_enable", True):
        return []
    red, blue = (float(g) for g in wb["colour_gains"])
    script = ("until ros2 param set /camera AwbEnable false; do sleep 2; done; "
              f"ros2 param set /camera ColourGains '[{red}, {blue}]'")
    return [ExecuteProcess(cmd=["bash", "-c", script], output="both")]


def person_follower(platform_text: str):
    """The platform's person follower, if its launch does not already start it."""
    if "person_follower" in platform_text:
        return []
    return [
        Node(package="hailo", executable="hailo_detection", output="both", respawn=True, respawn_delay=3.0),
        Node(package="person_follower", executable="person_follower_node", output="both"),
    ]


def generate_launch_description():
    platform_launch = Path(get_package_share_directory(PLATFORM_PACKAGE)) / "launch" / "launch.py"
    platform_text = platform_launch.read_text()
    has_mux = "cmd_vel_mux" in platform_text
    assistant = os.environ.get("SAR_ASSISTANT", "0") == "1"
    if has_mux:   # stock: the mux and the joystick behind it are already there
        platform = IncludeLaunchDescription(PythonLaunchDescriptionSource(str(platform_launch)))
        mux = []
    else:         # the course's variant: re-add the joystick, routed through our mux
        platform = IncludeLaunchDescription(PythonLaunchDescriptionSource(str(platform_launch)),
                                            launch_arguments={"teleop": "False"}.items())
        mux = [
            Node(package="teleop_twist_joy", executable="teleop_node", name="teleop_twist_joy_node",
                 parameters=[PLATFORM_CONFIG], remappings=[("cmd_vel", "teleop_cmd_vel")], output="both"),
            Node(package="cmd_vel_mux", executable="cmd_vel_mux_node", parameters=[PARAMS], output="both"),
        ]
    return LaunchDescription([
        platform,
        *mux,
        *camera_white_balance(),
        *(person_follower(platform_text) if assistant else []),
        ExecuteProcess(cmd=["python3", "-u", str(HERE.parent / "walk_controller.py"), *(["--tricks"] if assistant else [])],
                       output="both"),
    ])
