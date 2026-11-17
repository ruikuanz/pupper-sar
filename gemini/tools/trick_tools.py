"""Tricks: play_trick checks the request and queues the job; play_policy_trick is the job.

Every trick is played by a trained policy on its own controller (trick mode),
so the robot is always in one simple state. Two kinds of policy trick:

  playback     (trick.interactive is False), like sit: plays one reference
               motion from start to finish; its tracking policy keeps Pupper
               balanced while it plays.
  interactive  (trick.interactive is True), like sit_stay_shake: one Pupper
               stays in. It plays its opening clips, then loops one (the
               "stay") until Gemini asks for another (interactive_action), or
               any other motion leaves it through its exit clip.

A third kind, a reference trick (trick.kind == "reference"), is a joint-angle
table the running walking policy follows once, the way it jumps. Only the jump
is one, and it is opt-in: rename gemini/tricks/jump/trick.yaml.optional to
trick.yaml.
"""

from __future__ import annotations

from basics import PREPARES_MOVE
from following import prepare_to_move
from robot_api import FELL_MESSAGE, LOCOMOTION, LOCOMOTION_CONTROLLER, WALK_READY_S, Result, Robot, with_note
from tool_registry import TRICK_NAMES, Param, tool
from tricks import Trick


# ============================================================ what the robot does
async def play_trick(robot: Robot, name: str) -> Result:
    """Queue a trick from gemini/tricks. Gemini hears now that it was accepted."""
    trick = robot.tricks.get(name)
    if trick is None:
        return Result(False, f"unknown trick '{name}'; available: {list(robot.tricks)}")
    note = await prepare_to_move(robot)
    if trick.kind == "reference":
        job, then = (lambda: robot.play_reference_trick(trick)), LOCOMOTION
    else:
        job = lambda: play_policy_trick(robot, trick)  # noqa: E731
        # Where Pupper is once it is done: back walking after a playback trick that ends
        # standing (the job hands the legs back); otherwise in the trick (an interactive
        # trick stays in it; any other holds its end pose).
        then = LOCOMOTION if not trick.interactive and trick.end_pose == "standing" else trick
    return with_note(note, await robot.queue(f"trick {name}", job, trick.total_time, then=then))


async def play_policy_trick(robot: Robot, trick: Trick) -> Result:
    """The job that plays a policy trick, and leaves the robot in the right state, saying which.

    1. An interactive trick Pupper is already in: nothing to do.
    2. A playback trick still holding the end of its last run: relax first.
       Switching to a controller that is already on does nothing, so without
       this "swim again" would just keep lying there.
    3. Give the legs to the trick, and wait while it plays.
    4. Afterwards: an interactive trick stays (and one that walks in its own
       stance hands the legs to its walking policy); a playback trick that ends
       standing hands the legs back to the walking policy; any other holds its
       final pose.
    """
    if trick.interactive and robot.state_now is trick:  # 1
        return Result(True, f"already doing {trick.name}")
    if not trick.interactive and robot.state_now is trick:  # 2
        res = await robot.switch_to(None)
        if not res.ok:
            return res
    res = await robot.switch_to(trick.controller)  # 3
    if not res.ok:
        return res
    await robot.sleep(trick.play_time + 0.3)
    if robot.fallen:
        return Result(False, f"{trick.name} ended in a fall: {FELL_MESSAGE}")
    if trick.interactive:  # 4
        actions = ", ".join(trick.actions) or "none"
        leave = f" Any other motion first plays {trick.exit_clip}." if trick.exit_clip else ""
        if trick.walks:
            res = await robot.switch_to(trick.walk_controller)
            if not res.ok:
                return Result(False, f"played {trick.name}, but its walking policy did not take over: {res.message}")
            await robot.sleep(WALK_READY_S)  # it eases into the stance, then fades its policy in
            return Result(True, f"played {trick.name}; it now stays in its {trick.end_pose} stance."
                                f" {trick.walk.describe()} Interactive actions: {actions}.{leave}")
        return Result(True, f"played {trick.name}; it now stays {trick.end_pose}. Interactive actions: {actions}.{leave}")
    if trick.end_pose == "standing":
        res = await robot.switch_to(LOCOMOTION_CONTROLLER)
        if not res.ok:
            return Result(False, f"played {trick.name} but could not return to locomotion mode: {res.message}")
        return Result(True, f"played {trick.name}; back in locomotion mode")
    return Result(True, f"played {trick.name}; it holds its final pose ({trick.end_pose})")


# ============================================================ what Gemini reads
@tool(
    "play_trick",
    "Play one of Pupper's tricks. Each trick says below whether it starts standing or lying down and"
    " how it ends. A trick keeps its final pose until the next motion: after one that ends lying"
    " down, call stand_up before walking. Some tricks are interactive: Pupper stays in them and takes"
    " interactive_action requests until another motion. Only play tricks from this list." + PREPARES_MOVE
    + "\nTricks:\n{tricks}",
    name=Param("string", "The trick's name, exactly as in the list.", enum=TRICK_NAMES),
)
async def play_trick_tool(robot: Robot, name: str) -> Result:
    return await play_trick(robot, name)
