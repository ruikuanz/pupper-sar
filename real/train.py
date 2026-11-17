"""Teach Pupper the real game from a person's play.

    python3 real/train.py                          # every recording under real/recordings/ -> real/model.npz
    python3 real/train.py real/recordings/20261115-140502 --out real/model.npz

    python3 surfer/server.py --sim --game real --model real/model.npz    # then Pupper plays from it

The recordings come from `real/play.py` with "Record my play" on: the pictures
of each of the person's runs, and when which arrow key went down. This turns
them into examples of the form

    this picture (and the one a tenth of a game second before)  ->  the key the person pressed just then, or none

and fits the small network of surfer/real_learned.py to them. It needs PyTorch
(`pip install -r real/requirements-train.txt`), on the laptop only: the result
is a file of numbers that Pupper runs with numpy alone.

How a recording becomes examples (the numbers are at the top of the file):

    ... none | not counted | LEFT LEFT LEFT | not counted | none ...
                             ^ from 0.10 s before the key to just after it
    The pictures just outside that window look the same as the ones inside it,
    so they are not counted for or against: the network is not punished for
    pressing a moment early or late.
    The last second of every run is left out: it is the mistake that ended it.
    Every picture is also used mirrored, with left and right swapped: the
    game's rules are the same both ways round, and it doubles the lessons.

How to read the score it prints. One run in five is kept aside and never
trained on. On those runs the model is played picture by picture, the way
Pupper will use it, and its key presses are compared with the person's:

    made   the person pressed, and the model pressed the same key within 0.25 game s
    extra  the model pressed and the person did not

This tells you whether it learned what the person does. It does not tell you how
long Pupper survives: for that, let it play (real/README.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "surfer"))
import real_learned as L  # noqa: E402

try:
    import torch
    from torch import nn
except ImportError:  # the parts that only read recordings still work (tests/test_real.py uses them)
    torch = nn = None

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
MODEL_FILE = Path(__file__).resolve().parent / "model.npz"
# --- from a recording to examples (all in game seconds) ---------------------------
LABEL_BEFORE_S = 0.10      # pictures from this long before a key press ...
LABEL_AFTER_S = 0.02       # ... to this long after it are examples of that key
UNSURE_S = 0.20            # pictures this close to a key press, but outside that window, are not counted
DROP_END_S = 1.0           # the end of each run is the crash: not an example of anything
# --- training ---------------------------------------------------------------------
EPOCHS = 14
BATCH = 128
STEPS_MIN = 40             # batches per epoch, at least (a short recording is gone through more than once)
RATE = 1e-3
DROPOUT = 0.3
SHIFT_PX = 4               # each batch is slid sideways by up to this much, so a pixel's exact place matters less
BRIGHTNESS = 0.2           # ... and made up to this much brighter or darker
TINT = 0.1                 # ... with each colour up to this much stronger or weaker (the city's colours change)
KEEP_ASIDE_EVERY = 5       # one run in this many is kept for scoring
# --- scoring ----------------------------------------------------------------------
MATCH_S = 0.25             # the model's press counts as the person's if it is the same key within this
THRESHOLDS = np.arange(0.30, 0.91, 0.05)


@dataclass
class Run:
    name: str
    times: np.ndarray          # game seconds of each picture
    keys: list                 # [game seconds, move]
    pictures: np.ndarray       # N x 90 x 160 x 3, blue-green-red
    labels: np.ndarray         # index into L.MOVES for each picture
    counted: np.ndarray        # False: this picture is too close to call (or is the crash)
    earlier: np.ndarray        # index of the picture GAP_S before each picture


def label(times: np.ndarray, keys: list, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Which move each picture is an example of, and which pictures count at all."""
    labels = np.zeros(len(times), dtype=np.int64)
    counted = times <= seconds - DROP_END_S
    for at, move in keys:
        near = np.abs(times - at) <= UNSURE_S
        counted &= ~near
    for at, move in keys:
        if at > seconds - DROP_END_S:
            continue
        inside = (times >= at - LABEL_BEFORE_S) & (times <= at + LABEL_AFTER_S)
        labels[inside] = L.MOVES.index(move)
        counted |= inside
    counted &= times <= seconds - DROP_END_S
    return labels, counted


def load(folders: list[Path]) -> list[Run]:
    runs = []
    files = sorted(file for folder in folders for file in folder.rglob("run.json"))
    for file in files:
        info = json.loads(file.read_text())
        times = np.array(info["times"], dtype=np.float64)
        pictures = np.empty((len(times), L.INPUT_H, L.INPUT_W, 3), dtype=np.uint8)
        for index in range(len(times)):
            picture = cv2.imread(str(file.parent / f"{index:05d}.jpg"))
            if picture is None:
                raise SystemExit(f"{file.parent}: picture {index} is missing")
            pictures[index] = L.small(picture)
        labels, counted = label(times, info["keys"], info["seconds"])
        earlier = np.maximum(np.searchsorted(times, times - L.GAP_S, side="right") - 1, 0)
        runs.append(Run(f"{file.parent.parent.name}/{file.parent.name}", times, info["keys"], pictures, labels, counted, earlier))
    return runs


def build():
    """The network of real_learned.LAYERS, as PyTorch layers."""
    layers, channels, height, width = [], 6, L.INPUT_H, L.INPUT_W
    size = None
    for layer in L.LAYERS:
        if layer[0] == "conv":
            _, outputs, kernel, stride = layer
            layers += [nn.Conv2d(channels, outputs, kernel, stride, kernel // 2), nn.ReLU()]
            channels, height, width = outputs, -(-height // stride), -(-width // stride)
        else:
            if size is None:
                size = channels * height * width
                layers += [nn.Flatten(), nn.Dropout(DROPOUT)]
            layers += [nn.Linear(size, layer[1]), nn.ReLU()]
            size = layer[1]
    return nn.Sequential(*layers[:-1])   # no ReLU after the last layer


def export(net) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy()) for m in net if isinstance(m, (nn.Conv2d, nn.Linear))]


class Examples:
    """All the runs' pictures in one array, so a batch is a list of numbers into it."""

    def __init__(self, runs: list[Run], device) -> None:
        self.pictures = torch.from_numpy(np.concatenate([run.pictures for run in runs]))
        starts = np.cumsum([0] + [len(run.times) for run in runs[:-1]])
        self.earlier = torch.from_numpy(np.concatenate([run.earlier + start for run, start in zip(runs, starts)]))
        self.labels = torch.from_numpy(np.concatenate([run.labels for run in runs]))
        self.counted = np.flatnonzero(np.concatenate([run.counted for run in runs]))
        self.device = device

    def batch(self, indices: np.ndarray, vary: bool) -> tuple:
        indices = torch.from_numpy(indices)
        now, before = self.pictures[indices], self.pictures[self.earlier[indices]]
        x = torch.cat([now, before], dim=3).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        y = self.labels[indices].to(self.device)
        if vary:
            flip = torch.rand(len(indices), device=self.device) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
            left, right = L.MOVES.index("left"), L.MOVES.index("right")
            y = torch.where(flip & (y == left), right, torch.where(flip & (y == right), left, y))
            gain = 1 + BRIGHTNESS * (2 * torch.rand(len(indices), 1, 1, 1, device=self.device) - 1)
            tint = 1 + TINT * (2 * torch.rand(len(indices), 3, 1, 1, device=self.device) - 1)
            x = (x * gain * tint.repeat(1, 2, 1, 1)).clamp(0, 1)
            shift = int(np.random.randint(-SHIFT_PX, SHIFT_PX + 1))
            if shift:
                x = torch.roll(x, shift, dims=3)
        return x, y


def likely(net, run: Run, device) -> np.ndarray:
    """What the model says for every picture of a run, in order."""
    out = []
    net.eval()
    with torch.no_grad():
        for start in range(0, len(run.times), 256):
            now = torch.from_numpy(run.pictures[start:start + 256])
            before = torch.from_numpy(run.pictures[run.earlier[start:start + 256]])
            x = torch.cat([now, before], dim=3).to(device).permute(0, 3, 1, 2).float() / 255.0
            out.append(torch.softmax(net(x), dim=1).cpu().numpy())
    return np.concatenate(out)


def presses(times: np.ndarray, probabilities: np.ndarray, threshold: float) -> list:
    """Play a run the way real_learned.LearnedPilot would: [game seconds, move] of every press."""
    out, busy_until = [], 0.0
    for at, row in zip(times, probabilities):
        move = L.choose(row, at, busy_until, threshold)
        if move:
            out.append([at, move])
            busy_until = at + L.REFRACTORY_S
    return out


def compare(person: list, model: list, seconds: float) -> tuple[int, int, int]:
    """(made, missed, extra): the model's presses against the person's, up to the crash."""
    person = [key for key in person if key[0] <= seconds - DROP_END_S]
    model = [key for key in model if key[0] <= seconds - DROP_END_S]
    used, made = set(), 0
    for at, move in person:
        for index, (model_at, model_move) in enumerate(model):
            if index not in used and model_move == move and abs(model_at - at) <= MATCH_S:
                used.add(index)
                made += 1
                break
    return made, len(person) - made, len(model) - len(used)


def score(net, runs: list[Run], device) -> tuple[float, dict]:
    """The best threshold on these runs, and how the model does there."""
    probabilities = [likely(net, run, device) for run in runs]
    minutes = sum(max(run.times[-1] - DROP_END_S, 0.0) for run in runs) / 60 or 1e-9
    best = None
    for threshold in THRESHOLDS:
        made = missed = extra = 0
        for run, rows in zip(runs, probabilities):
            a, b, c = compare(run.keys, presses(run.times, rows, threshold), run.times[-1])
            made, missed, extra = made + a, missed + b, extra + c
        both = 2 * made / max(2 * made + missed + extra, 1)
        if best is None or both > best[0]:
            best = (both, float(threshold), {"made": made, "missed": missed, "extra": extra,
                                              "extra_per_game_minute": round(float(extra / minutes), 1)})
    return best[1], best[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Teach Pupper the real game from a person's recorded play.")
    parser.add_argument("recordings", nargs="*", type=Path, default=[RECORDINGS_DIR], help="folders of recordings (default: real/recordings)")
    parser.add_argument("--out", type=Path, default=MODEL_FILE, help="the model file to write (default: real/model.npz)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if torch is None:
        raise SystemExit("Training needs PyTorch: pip install -r real/requirements-train.txt")

    teach(args.recordings, args.out, args.epochs)


def teach(recordings: list[Path], out: Path, epochs: int = EPOCHS) -> dict:
    """Recordings -> a model file. Returns its score on the kept-aside runs."""
    started = time.time()
    runs = load(recordings)
    if not runs:
        raise SystemExit(f"no recordings under {', '.join(map(str, recordings))}: play with \"Record my play\" on (real/play.py)")
    if len(runs) >= KEEP_ASIDE_EVERY:
        aside = [run for index, run in enumerate(runs) if index % KEEP_ASIDE_EVERY == KEEP_ASIDE_EVERY - 1]
        taught = [run for index, run in enumerate(runs) if index % KEEP_ASIDE_EVERY != KEEP_ASIDE_EVERY - 1]
    else:
        aside, taught = runs, runs
        print(f"only {len(runs)} runs: nothing can be kept aside, so the score below is on runs it was taught (too rosy)")
    minutes = sum(run.times[-1] for run in runs) / 60
    counts = np.bincount(np.concatenate([run.labels[run.counted] for run in taught]), minlength=len(L.MOVES))
    print(f"{len(runs)} runs, {minutes:.1f} game minutes, {sum(len(run.keys) for run in runs)} key presses")
    print("examples: " + ", ".join(f"{move} {count}" for move, count in zip(L.MOVES, counts)))
    if counts[1:].min() < 30:
        print("that is very few examples of some moves: expect little. An hour of play is the aim.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    examples = Examples(taught, device)
    torch.manual_seed(0)
    np.random.seed(0)
    net = build().to(device)
    # Moves are rare next to "none": show them more often, but not so often that the network cries wolf.
    labels = examples.labels.numpy()[examples.counted]
    weight = (len(labels) / np.maximum(counts, 1)) ** 0.5
    chance = weight[labels] / weight[labels].sum()
    steps = max(len(labels) // BATCH, STEPS_MIN)
    optimiser = torch.optim.AdamW(net.parameters(), lr=RATE, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=RATE, total_steps=epochs * steps)
    for epoch in range(epochs):
        net.train()
        total = 0.0
        for _ in range(steps):
            x, y = examples.batch(np.sort(examples.counted[np.random.choice(len(labels), BATCH, p=chance)]), vary=True)
            loss = nn.functional.cross_entropy(net(x), y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item()
        threshold, result = score(net, aside, device)
        print(f"epoch {epoch + 1:2d}/{epochs}  loss {total / steps:.3f}  kept-aside runs: made {result['made']}, "
              f"missed {result['missed']}, extra {result['extra']} (threshold {threshold:.2f})", flush=True)

    threshold, result = score(net, aside, device)
    L.save(out, export(net), {
        "threshold": threshold, "runs": len(runs), "game_minutes": round(minutes, 1), "score": result,
        "scored_on_taught_runs": aside is taught, "trained": time.strftime("%Y-%m-%d %H:%M")})
    # The file must give the same answers through numpy (what Pupper runs) as through PyTorch.
    check = aside[0]
    numpy_net = L.Net(out)
    difference = float(np.abs(numpy_net(L.pair(check.pictures[-1], check.pictures[check.earlier[-1]])) - likely(net, check, device)[-1]).max())
    person = result["made"] + result["missed"]
    print(f"\nwrote {out} in {time.time() - started:.0f} s (numpy and PyTorch agree to {difference:.1e})")
    print(f"on the kept-aside runs it made {result['made']} of the person's {person} moves, and {result['extra']} extra "
          f"({result['extra_per_game_minute']} per game minute), pressing when a move is at least {threshold:.2f} likely")
    print(f"next: python3 surfer/server.py --sim --game real --model {out}")
    return {**result, "threshold": threshold, "agreement": difference}


if __name__ == "__main__":
    main()
