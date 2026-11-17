"""Follow mode: Pupper walks after the nearest person, and the rule that ends it.

The person follower itself runs in the robot stack (the platform's Hailo person
detector and follower; robot.set_following turns it on and off). What lives here
is the rule:

    Questions and conversation keep following. Any motion (walk, turn, arc,
    trick, stand up, lie down) ends follow mode right away, and the reply says
    so, so Gemini can tell the person: "Let me stop following you and ..."

Why right away, and not when the queue gets to the motion: the person asked for
something else *now*. Why in code and not only in the prompt: a model can
forget a sentence; a function every motion calls cannot.
"""

from __future__ import annotations

from robot_api import Result, Robot
from tool_registry import Param, tool


# ============================================================ what the robot does
async def prepare_to_move(robot: Robot) -> str:
    """Get ready for a new motion. Every motion calls this first, before it queues anything.

    It stops whatever runs on its own outside the queue and would fight the new
    motion. Today that is the person follower; anything like it added later (a
    patrol loop, dancing to music) belongs here too. Getting the legs ready is a
    different job, done later by switch_to.

    Returns a note for the front of the tool's reply ("" if nothing was stopped).
    """
    if not robot.following:
        return ""
    await robot.stop_following()
    robot.send_speed(0, 0, 0)  # the follower may have been walking
    return "Stopped following the person first."


# ============================================================ what Gemini reads
@tool(
    "follow",
    "Turn follow mode on or off. On: Pupper stands up if needed and keeps walking after the nearest"
    " person, keeping its distance, until follow mode ends. Talking and questions (robot_status) keep"
    " following. Any motion tool (stand_up, lie_down, move, turn, arc, play_trick, interactive_action)"
    " ends it at once, and so does stop. Before you call one of those while following, tell the"
    " person you will stop following them.",
    enabled=Param("boolean", "true to start following the nearest person, false to stop following."),
)
async def follow_tool(robot: Robot, enabled: bool) -> Result:
    return await robot.set_following(enabled)
