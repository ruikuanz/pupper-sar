"""Pupper's learned eyes for the real game: a small network trained on a person's play.

    picture of the game (and the one a tenth of a second before)
        -> how likely is it that a person would press left / right / jump / squat right now?

real_perception.py reads the real game with hand-set colour rules, and the game
defeats them within seconds (real/README.md has the numbers). This file is the
other approach, the one a person detector takes and this takes for a whole game:
do not describe what to look for, show examples. The examples are a person
playing the left window (real/play.py records the pictures and the keys), and
real/train.py turns them into the file of numbers this file loads.

Only numpy and OpenCV are needed here, so it runs on Pupper as it is. The
training (real/train.py) is done on the laptop and needs PyTorch.

The network, which train.py must build identically (LAYERS):

    2 pictures, 160 x 90, blue-green-red each      6 x 90 x 160
    4 convolutions, each halving the picture       64 x 6 x 10
    2 ordinary layers                               5 numbers: none, left, right, jump, squat
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

MOVES = ("none", "left", "right", "jump", "squat")
INPUT_W, INPUT_H = 160, 90     # the pictures are shrunk to this before the network sees them
GAP_S = 0.10                   # the second picture is this much game time older: it shows what is moving
# (kind, outputs, kernel, stride): 3 = the size of the patch a convolution looks at, 2 = it halves the picture.
LAYERS = (("conv", 24, 5, 2), ("conv", 32, 3, 2), ("conv", 48, 3, 2), ("conv", 64, 3, 2), ("full", 128), ("full", len(MOVES)))
THRESHOLD = 0.5                # press when a move is at least this likely (train.py picks a better value)
REFRACTORY_S = 0.25            # after a press, no other press for this much game time: the runner is mid-move
HISTORY_S = 0.5                # how much game time of pictures to keep, for the older one of the pair


def small(view_bgr: np.ndarray) -> np.ndarray:
    """A picture of the game at the size the network was trained on."""
    if view_bgr.shape[1] == INPUT_W and view_bgr.shape[0] == INPUT_H:
        return view_bgr
    return cv2.resize(view_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)


def pair(now_small: np.ndarray, earlier_small: np.ndarray) -> np.ndarray:
    """The network's input: this picture and the earlier one, as 6 x 90 x 160 numbers from 0 to 1."""
    both = np.concatenate([now_small, earlier_small], axis=2)
    return np.ascontiguousarray(both.transpose(2, 0, 1), dtype=np.float32) / 255.0


def convolve(x: np.ndarray, weights: np.ndarray, bias: np.ndarray, stride: int) -> np.ndarray:
    """One convolution: x is channels x height x width, weights is outputs x channels x k x k."""
    k = weights.shape[2]
    x = np.pad(x, ((0, 0), (k // 2, k // 2), (k // 2, k // 2)))
    patches = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(1, 2))[:, ::stride, ::stride]
    return np.tensordot(weights, patches, axes=([1, 2, 3], [0, 3, 4])) + bias[:, None, None]


class Net:
    """The trained numbers, and the arithmetic that applies them to a picture pair."""

    def __init__(self, path: str | Path) -> None:
        with np.load(path, allow_pickle=False) as file:
            self.meta = json.loads(str(file["meta"]))
            self.layers = []
            for index, layer in enumerate(LAYERS):
                self.layers.append((layer, file[f"w{index}"].astype(np.float32), file[f"b{index}"].astype(np.float32)))
        if tuple(self.meta["moves"]) != MOVES or self.meta["input"] != [INPUT_W, INPUT_H]:
            raise ValueError(f"{path} was trained for a different network than this file describes")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Picture pair (6 x 90 x 160) -> how likely each of MOVES is (5 numbers that add up to 1)."""
        last = len(self.layers) - 1
        for index, (layer, weights, bias) in enumerate(self.layers):
            if layer[0] == "conv":
                x = convolve(x, weights, bias, layer[3])
            else:
                x = weights @ x.reshape(-1) + bias
            if index < last:
                x = np.maximum(x, 0.0)
        x = np.exp(x - x.max())
        return x / x.sum()


def save(path: str | Path, arrays: list[tuple[np.ndarray, np.ndarray]], meta: dict) -> None:
    """Write a model file: the (weights, bias) of each of LAYERS, and a note about how it was trained."""
    meta = {**meta, "moves": list(MOVES), "input": [INPUT_W, INPUT_H], "gap_s": GAP_S}
    numbers = {"meta": np.array(json.dumps(meta))}
    for index, (weights, bias) in enumerate(arrays):
        numbers[f"w{index}"], numbers[f"b{index}"] = weights.astype(np.float32), bias.astype(np.float32)
    np.savez(path, **numbers)


def choose(likely: np.ndarray, now: float, busy_until: float, threshold: float) -> str | None:
    """Which key to press on this picture, if any. Also used by train.py to score a model
    the way it will be used."""
    if now < busy_until:
        return None
    best = int(np.argmax(likely[1:])) + 1
    return MOVES[best] if likely[best] >= threshold else None


class LearnedPilot:
    """Plays from pictures: the same step() as real_policy.Pilot, but it takes the picture itself."""

    def __init__(self, path: str | Path) -> None:
        self.net = Net(path)
        self.threshold = float(self.net.meta.get("threshold", THRESHOLD))
        self.reset()

    def reset(self) -> None:
        self.lane = 1                  # where the runner should be, counted from the presses (for the status line)
        self.speed = 1.0               # kept so the status line reads the same as with real_policy.Pilot
        self.busy_until = 0.0
        self.history: deque = deque()  # (game time, small picture)
        self.likely = np.array([1.0, 0, 0, 0, 0], dtype=np.float32)

    def step(self, view_bgr: np.ndarray, now: float) -> str | None:
        """One picture at game time `now` (seconds) -> "left", "right", "jump", "squat" or None."""
        picture = small(view_bgr)
        history = self.history
        if history and now < history[-1][0]:
            history.clear()            # the clock went backwards: a new run
        history.append((now, picture))
        while len(history) > 1 and history[1][0] <= now - GAP_S:
            history.popleft()          # history[0] is now the newest picture that is at least GAP_S old
        while history and now - history[0][0] > HISTORY_S:
            history.popleft()
        earlier = history[0][1] if history else picture
        self.likely = self.net(pair(picture, earlier))
        move = choose(self.likely, now, self.busy_until, self.threshold)
        if move:
            self.busy_until = now + REFRACTORY_S
            self.lane = min(2, max(0, self.lane + {"left": -1, "right": 1}.get(move, 0)))
        return move


def draw_debug(view_bgr: np.ndarray, likely: np.ndarray, threshold: float, note: str = "") -> np.ndarray:
    """The picture with a bar for each move: how likely the network thinks a press is."""
    out = cv2.resize(view_bgr, (640, 360), interpolation=cv2.INTER_AREA) if view_bgr.shape[1] != 640 else view_bgr.copy()
    for index, move in enumerate(MOVES[1:], start=1):
        x, y = 8, 44 + 18 * index
        width = int(120 * float(likely[index]))
        colour = (60, 255, 60) if likely[index] >= threshold else (200, 200, 200)
        cv2.rectangle(out, (x + 50, y - 10), (x + 50 + 120, y), (40, 40, 40), -1)
        cv2.rectangle(out, (x + 50, y - 10), (x + 50 + width, y), colour, -1)
        cv2.putText(out, move, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    mark = x + 50 + int(120 * threshold)
    cv2.line(out, (mark, 50), (mark, 44 + 18 * 4 + 2), (0, 200, 255), 1)
    if note:
        cv2.putText(out, note, (8, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
