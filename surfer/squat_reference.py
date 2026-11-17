"""Build the squat: a reference table the walking policy plays like its jump.

    python3 surfer/squat_reference.py            # writes surfer/references/squat.json
    python3 surfer/squat_reference.py --drop 0.04 --hold 0.5

Pupper jumps by feeding its running walking policy a table of joint angles
(playback_s, active_s, cross_fade_s, then n rows of 12 angles) on
/sar_walk/reference_trick; the policy follows it and keeps its balance. A
squat is the same kind of table: ease down to a crouch, hold, ease back up.

The crouch is forward kinematics of each leg, and inverse
kinematics that moves every foot straight up toward the body by DROP_M while
keeping it under the same spot. Feet that don't slide are what keep the body
level and in its lane.

.. warning::
   The policy was TRAINED on its jump table, not on this one. The crouch is a
   pose the policy has been through (the jump's wind-up goes much lower: the
   feet come up about 10 cm there), but holding it is new to it. Test it the
   way any new motion is tested: on the ground, clear space, a hand on the
   e-stop, and start shallow (--drop 0.03). If the policy won't follow it, the
   proper fix is to train a tracking policy on this motion.

The leg geometry is the Pupper v3 MuJoCo model's (pupperv3-monorepo,
pupper_v3_description/description/mujoco_xml/pupper_v3_complete.xml): each
joint turns about its own z axis, and the feet are the model's foot sites.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "references" / "squat.json"

DROP_M = 0.045      # how far the body comes down (the hips stand 14 cm above the feet)
EASE_S = 0.25       # going down, and coming back up
HOLD_S = 0.50       # staying down
CROSS_FADE_S = 0.06  # the controller's blend at each end of the table (the jump's value)
SAMPLES = 100

# In the policy's joint order: front right, front left, back right, back left.
LEGS = ["front_r", "front_l", "back_r", "back_l"]
# Standing: the policy's default pose (default_policy.json, default_joint_pos).
STANDING = np.array([0.26, 0.0, -0.52, -0.26, 0.0, 0.52, 0.26, 0.0, -0.52, -0.26, 0.0, 0.52])
JOINT_LOW = np.array([-1.22, -0.42, -2.79, -2.51, -3.14, -0.71] * 2)
JOINT_HIGH = np.array([2.51, 3.14, 0.71, 1.22, 0.42, 2.79] * 2)

S = 0.7071068
# For each leg: (position, quaternion w x y z) of joints 1, 2, 3 in their parent, and the foot in link 3.
GEOMETRY = {
    "front_r": ([((0.075, -0.0835, 0.0), (S, S, 0, 0)), ((0, 0, 0), (S, 0, -S, 0)), ((0, -0.0494, 0.0685), (S, 0, S, 0))],
                (0.06231, -0.06216, 0.018)),
    "front_l": ([((0.075, 0.0835, 0.0), (S, -S, 0, 0)), ((0, 0, 0), (0, S, 0, -S)), ((0, -0.0494, 0.0685), (0, -S, 0, S))],
                (0.06231, 0.06216, 0.018)),
    "back_r": ([((-0.075, -0.0725, 0.0), (S, S, 0, 0)), ((0, 0, 0), (S, 0, -S, 0)), ((0, -0.0494, 0.0685), (S, 0, S, 0))],
               (0.06231, -0.06216, 0.018)),
    "back_l": ([((-0.075, 0.0725, 0.0), (S, -S, 0, 0)), ((0, 0, 0), (0, S, 0, -S)), ((0, -0.0494, 0.0685), (0, -S, 0, S))],
               (0.06231, 0.06216, 0.018)),
}


def rotation_from_quaternion(q) -> np.ndarray:
    w, x, y, z = np.array(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def transform(rotation: np.ndarray, position) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rotation
    m[:3, 3] = position
    return m


def foot_position(leg: str, angles) -> np.ndarray:
    """Forward kinematics: the foot in the body frame (m), for one leg's three joint angles."""
    joints, foot = GEOMETRY[leg]
    m = np.eye(4)
    for (position, quaternion), angle in zip(joints, angles):
        m = m @ transform(rotation_from_quaternion(quaternion), position) @ transform(rotation_z(angle), (0, 0, 0))
    return (m @ np.array([*foot, 1.0]))[:3]


def inverse_kinematics(leg: str, target, guess, iterations: int = 200, tolerance: float = 1e-6) -> np.ndarray:
    """Joint angles that put the foot at target: Newton steps on a finite-difference Jacobian."""
    angles = np.array(guess, dtype=float)
    for _ in range(iterations):
        error = foot_position(leg, angles) - target
        if np.linalg.norm(error) < tolerance:
            break
        jacobian = np.zeros((3, 3))
        for k in range(3):
            step = np.zeros(3)
            step[k] = 1e-5
            jacobian[:, k] = (foot_position(leg, angles + step) - foot_position(leg, angles - step)) / 2e-5
        angles -= np.linalg.solve(jacobian.T @ jacobian + 1e-8 * np.eye(3), jacobian.T @ error)
    return angles


def crouch_pose(drop: float) -> np.ndarray:
    """The 12 joint angles with every foot `drop` m closer to the body than standing."""
    pose = []
    for i, leg in enumerate(LEGS):
        standing = STANDING[3 * i:3 * i + 3]
        target = foot_position(leg, standing) + np.array([0.0, 0.0, drop])
        angles = inverse_kinematics(leg, target, standing)
        miss = np.linalg.norm(foot_position(leg, angles) - target)
        if miss > 1e-3:
            raise ValueError(f"{leg} can't reach a {drop} m crouch (off by {1000 * miss:.1f} mm)")
        pose.extend(angles)
    pose = np.array(pose)
    if np.any(pose < JOINT_LOW) or np.any(pose > JOINT_HIGH):
        raise ValueError(f"a {drop} m crouch is outside the joint limits")
    return pose


def build(drop: float = DROP_M, ease: float = EASE_S, hold: float = HOLD_S) -> dict:
    """The reference: stand -> crouch (ease) -> stay (hold) -> stand (ease)."""
    total = 2 * ease + hold
    table = []
    for i in range(SAMPLES):
        t = total * i / (SAMPLES - 1)
        down = min(t / ease, (total - t) / ease, 1.0)      # 0 standing .. 1 fully down
        smooth = down * down * (3 - 2 * down)
        # IK at every sample (not a blend of joint angles): the feet move straight up and down.
        table.append([round(float(a), 5) for a in crouch_pose(smooth * drop)])
    return {
        "about": "A squat for the default walking policy: joint angles (the policy's joint order), "
                 "played once over playback_s through the jump slot. Built by surfer/squat_reference.py.",
        "drop_m": drop,
        "playback_s": total,
        "active_s": total,
        "cross_fade_s": CROSS_FADE_S,
        "table": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--drop", type=float, default=DROP_M, help="m the body comes down (default %(default)s)")
    parser.add_argument("--ease", type=float, default=EASE_S, help="s going down, and coming up")
    parser.add_argument("--hold", type=float, default=HOLD_S, help="s staying down")
    args = parser.parse_args()
    reference = build(args.drop, args.ease, args.hold)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(reference))
    low = np.array(reference["table"][SAMPLES // 2])
    print(f"wrote {OUT}: {reference['playback_s']:.2f} s, body down {100 * args.drop:.1f} cm")
    print("standing :", np.round(STANDING, 2).tolist())
    print("crouched :", np.round(low, 2).tolist())
    for i, leg in enumerate(LEGS):
        print(f"  {leg:8s} foot {np.round(foot_position(leg, STANDING[3 * i:3 * i + 3]), 3).tolist()}"
              f" -> {np.round(foot_position(leg, low[3 * i:3 * i + 3]), 3).tolist()}")


if __name__ == "__main__":
    main()
