"""Interactive actions: the clips Gemini asks for while Pupper stays in an interactive trick.

An interactive trick (trick_tools.py explains them) is trained on a clip
library; its trick.yaml names the clips Gemini may ask for (actions) and the
one that stands back up (exit_clip). No interactive trick ships in
gemini/tricks yet; one fetched with gemini/get_trick.py shows up here by itself.
"""

from __future__ import annotations

from robot_api import LOCOMOTION, LOCOMOTION_CONTROLLER, Result, Robot
from tool_registry import ACTION_IDS, Param, tool


# ============================================================ what the robot does
async def interactive_action(robot: Robot, action: str) -> Result:
    """Play one of an interactive trick's actions (e.g. shake a paw while sitting).

    Checked twice. Now, against robot.state_after_queue: "sit, then shake" said
    in one breath queues both before the sit has happened, and the shake must be
    accepted. Later, in the job, against robot.state_now: Pupper may have fallen,
    been e-stopped or stopped in between.
    """
    found = robot.action_ids().get(action)
    if found is None:
        return Result(False, f"unknown interactive action {action}; available: {list(robot.action_ids())}")
    trick, clip = found
    if robot.state_after_queue is not trick:
        return Result(False, f"{action} only works while {trick.name} is running; play it first (play_trick {trick.name})")
    leaving = clip == trick.exit_clip

    async def job() -> Result:
        if robot.state_now is not trick:
            return Result(False, f"{action} not played: {trick.name} is no longer running (play_trick {trick.name} first)")
        if leaving:  # switching back to walking plays the exit clip on the way out
            res = await robot.switch_to(LOCOMOTION_CONTROLLER)
            return Result(res.ok, f"played {clip}; back in locomotion mode" if res.ok else res.message)
        return await robot.play_clip(trick, clip)

    return await robot.queue(f"{trick.name} {clip}", job, trick.clip_time_estimate(clip),
                             then=LOCOMOTION if leaving else trick)


# ============================================================ what Gemini reads
@tool(
    "interactive_action",
    "Do one of the actions of an interactive trick, while Pupper is in that trick (play_trick it first;"
    " both can be called back to back). Pupper stays in the trick afterwards, except for an action that"
    " stands back up. Any other motion tool also leaves the trick, through its stand-up clip.\n"
    "Actions:\n{actions}",
    action=Param("string", "The action's name, exactly as in the list.", enum=ACTION_IDS),
)
async def interactive_action_tool(robot: Robot, action: str) -> Result:
    return await interactive_action(robot, action)
