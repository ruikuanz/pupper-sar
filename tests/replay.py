"""Replay whole games through Pupper's vision, faster than real time.

    python3 tests/replay.py                 # seeds 1-6, the first 60 s of each
    python3 tests/replay.py --seeds 9 12 --seconds 120

For each seed it renders Pupper's view of the course 15 times a second with
the real game renderer (tests/render_frames.mjs, so this needs `npm install`
in tests/), runs every frame through perception.detect and RowTracker, and
compares the rows the tracker believes in with the rows that are really there.
It is how the "phantom wall" bug was found: one crash in a 90 s game became
one line of this output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "surfer"))
import perception as P  # noqa: E402

LETTER = {P.FREE: ".", P.HURDLE: "J", P.BEAM: "S", P.WALL: "X"}
Z_MAX = 22.0  # compare rows nearer than this. Rows are 24 apart, so this is always the one row
              # nothing can hide; a row farther off may be behind a wall of the row before it.


def short(cells) -> str:
    return "".join(LETTER[c] for c in cells)


def replay(seed: int, seconds: float, lane: int) -> tuple[int, int]:
    with tempfile.TemporaryDirectory() as out:
        subprocess.run(["node", str(ROOT / "tests" / "render_frames.mjs"), "--seed", str(seed), "--to", str(seconds),
                        "--lane", str(lane), "--out", out], check=True, capture_output=True)
        tracker, wrong, frames = P.RowTracker(), 0, 0
        for path in sorted(Path(out).glob("*.json")):
            truth = json.loads(path.read_text())
            rows = tracker.update(P.detect(cv2.imread(str(path.with_suffix(".png")))), truth["t"], (truth["lane"],))
            want = [(r["z"], short(r["cells"])) for r in truth["rows"]]
            got = [(round(r.z, 1), short(r.cells)) for r in rows]
            # Every real row in range must be believed, and every believed row must be real.
            in_range = lambda z: 0.7 <= z <= Z_MAX
            matches = lambda a, others: any(abs(a[0] - o[0]) <= max(1.0, 0.1 * a[0]) and a[1] == o[1] for o in others)
            same = all(matches(w, got) for w in want if in_range(w[0])) and all(matches(g, want) for g in got if in_range(g[0]))
            frames += 1
            if not same:
                wrong += 1
                if wrong <= 3:
                    print(f"    seed {seed} t={truth['t']:.2f}: truth {want}  tracker {got}")
        return wrong, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()
    total_wrong = 0
    for seed in args.seeds:
        for lane in (0, 1, 2):
            wrong, frames = replay(seed, args.seconds, lane)
            total_wrong += wrong
            print(f"  {'ok  ' if wrong == 0 else 'FAIL'}  seed {seed}, runner in lane {lane}: {frames - wrong} of {frames} frames read correctly")
    print("PASSED" if total_wrong == 0 else f"FAILED: {total_wrong} frames with a wrong row")
    sys.exit(1 if total_wrong else 0)


if __name__ == "__main__":
    main()
