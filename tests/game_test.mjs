// Tests of the game's rules and of the body-pose controls. No browser needed:
//
//   node tests/game_test.mjs

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { BEAM, Course, FREE, Game, HURDLE, WALL, reachable } from "../game/js/engine.js";
import { PoseMoves } from "../game/js/pose_moves.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const config = JSON.parse(readFileSync(join(ROOT, "game/config.json"), "utf8"));
let failed = 0, count = 0;

function check(name, ok, got, want) {
  count += 1;
  if (ok) {
    console.log(`  ok    ${name}`);
  } else {
    failed += 1;
    console.log(`  FAIL  ${name}\n          got:    ${JSON.stringify(got)}\n          wanted: ${JSON.stringify(want)}`);
  }
}

// ------------------------------------------------------------------ the course
console.log("the course");
{
  const a = new Course(7, config.presets.pupper), b = new Course(7, config.presets.pupper), c = new Course(8, config.presets.pupper);
  const rows = (course) => JSON.stringify(Array.from({ length: 40 }, (_, i) => course.row(i).cells));
  check("the same seed gives the same course (the two views are replicas)", rows(a) === rows(b), "different", "same");
  check("another seed gives another course", rows(a) !== rows(c), "same", "different");

  let unfair = null, kinds = new Set();
  for (let seed = 1; seed <= 30 && !unfair; seed++) {
    const course = new Course(seed, config.presets.pupper);
    for (let i = 0; i < 200; i++) {
      const row = course.row(i);
      row.cells.forEach((cell) => kinds.add(cell));
      const prev = i ? course.row(i - 1).cells : [FREE, FREE, FREE];
      if (!reachable(prev, row.cells) || row.cells.every((cell) => cell === WALL)) unfair = { seed, i, prev, cells: row.cells };
    }
  }
  check("every row can be passed by changing at most one lane since the row before (30 seeds x 200 rows)", !unfair, unfair, null);
  check("hurdles, beams, walls and free lanes all appear", kinds.size === 4, [...kinds], [FREE, HURDLE, BEAM, WALL]);
  const first = Array.from({ length: 4 }, (_, i) => a.row(i).cells.join(" "));
  check("the first rows make a demo use every move: jump, squat, then two lane changes",
    first[0] === "hurdle hurdle hurdle" && first[1] === "beam beam beam" && first[2] === "free wall free" && first[3] === "wall free wall", first, "the intro");
}

// ------------------------------------------------------------------ the rules
console.log("the rules");
{
  // Play one player through the intro with moves made at chosen times.
  const play = (script, seconds = 16) => {
    const game = new Game(config, { seed: 1, preset: "pupper" });
    game.human.endless = true;
    game.start();
    const events = [];
    const todo = [...script];
    for (let t = 0; t < seconds + config.timing.countdown_s; t += 1 / 60) {
      while (todo.length && game.playTime >= todo[0][0] && game.state === "running") {
        const [, move, opts] = todo.shift();
        game.pupper.command(move, game.t, opts);
      }
      for (const e of game.update(1 / 60)) if (e.player === "pupper") events.push(e);
    }
    return { game, crashes: events.filter((e) => e.type === "crash"), passes: events.filter((e) => e.type === "pass") };
  };
  // Rows arrive at 4, 7, 10, 13 s: hurdles, beams, a wall in the centre, walls at the sides.
  const idle = play([]);
  check("doing nothing crashes into the first hurdle, and costs a life", idle.crashes[0]?.cell === HURDLE && idle.game.pupper.lives < 3, idle.crashes[0], "a crash into a hurdle");
  const good = play([[3.65, "jump"], [6.65, "squat"], [8, "left"], [11, "right"]]);
  check("jump, squat, left, right at the right times: no crash", good.crashes.length === 0 && good.passes.length >= 4, { crashes: good.crashes, passes: good.passes.length }, "no crashes, 4 rows passed");
  const early = play([[2.5, "jump"]], 5);
  check("a jump that has landed before the hurdle arrives does not count", early.crashes.length === 1, early.crashes.length, 1);
  const wrong = play([[3.65, "squat"]], 5);
  check("squatting does not clear a hurdle", wrong.crashes.length === 1, wrong.crashes.length, 1);
  const slow = play([[3.25, "jump", { duration: 0.94 }], [6.5, "squat", { duration: 1.0 }], [8, "left", { duration: 0.8 }], [11, "right", { duration: 0.8 }]]);
  check("Pupper's slower moves (0.94 s jump, 0.8 s shuffle) clear the same course", slow.crashes.length === 0, slow.crashes, []);
  const late = play([[3.65, "jump"], [6.65, "squat"], [9.9, "left", { duration: 0.8 }]], 11);
  check("a shuffle started too late is still in the old lane when the wall arrives", late.crashes[0]?.cell === WALL, late.crashes[0], "a crash into the wall");
  const edge = new Game(config, { seed: 1 });
  edge.start();
  edge.pupper.command("left", 0);
  check("no lane beyond the left lane", edge.pupper.command("left", 0) === false && edge.pupper.lane === 0, edge.pupper.lane, 0);
  const over = play([], 40);
  check("three crashes end the run", !over.game.pupper.alive && over.game.pupper.crashes === 3, over.game.pupper.crashes, 3);
}

// ------------------------------------------------------------------ body pose
console.log("body pose -> moves");
{
  // A stick figure: shoulders and hips, in the camera's (unmirrored) picture.
  const body = ({ x = 0.5, up = 0, crouch = 0 } = {}) => {
    const lm = Array.from({ length: 33 }, () => ({ x: 0, y: 0, visibility: 0 }));
    const shoulderY = 0.35 - up + crouch, hipY = 0.6 - up + 0.3 * crouch;
    // The person's LEFT shoulder (11) is on the picture's right.
    lm[11] = { x: x + 0.08, y: shoulderY, visibility: 0.99 };
    lm[12] = { x: x - 0.08, y: shoulderY, visibility: 0.99 };
    lm[23] = { x: x + 0.06, y: hipY, visibility: 0.99 };
    lm[24] = { x: x - 0.06, y: hipY, visibility: 0.99 };
    return lm;
  };
  const calibrated = () => {
    const pose = new PoseMoves();
    pose.startCalibration(0);
    for (let t = 0; t <= 1.6; t += 1 / 30) pose.update(body(), t);
    return pose;
  };
  const feed = (pose, frames) => frames.flatMap((f, i) => pose.update(body(f), 2 + i / 30));

  check("standing still after calibration: no moves", feed(calibrated(), Array(30).fill({})).length === 0, "moves", "none");
  // A step to the person's right moves them toward the picture's LEFT (smaller x).
  let got = feed(calibrated(), [{ x: 0.45 }, { x: 0.4 }, { x: 0.35 }, { x: 0.35 }]);
  check("a step to the person's right (picture left) is one move right", JSON.stringify(got) === '["right"]', got, ["right"]);
  got = feed(calibrated(), [{ x: 0.65 }, { x: 0.65 }]);
  check("a step to the person's left is one move left", JSON.stringify(got) === '["left"]', got, ["left"]);
  got = feed(calibrated(), [{ x: 0.35 }, { x: 0.36 }, { x: 0.37 }, { x: 0.36 }, { x: 0.38 }]);
  check("swaying at the edge of a lane does not flicker", JSON.stringify(got) === '["right"]', got, ["right"]);
  got = feed(calibrated(), [{ x: 0.35 }, { x: 0.5 }, { x: 0.65 }]);
  check("right, back to the centre, then left: right, left, left", JSON.stringify(got) === '["right","left","left"]', got, ["right", "left", "left"]);
  got = feed(calibrated(), [{ up: 0.02 }, { up: 0.06 }, { up: 0.07 }, { up: 0.03 }, {}, {}]);
  check("a jump is one jump", JSON.stringify(got) === '["jump"]', got, ["jump"]);
  got = feed(calibrated(), [{ crouch: 0.05 }, { crouch: 0.12 }, { crouch: 0.12 }, { crouch: 0.05 }, {}]);
  check("a squat is one squat", JSON.stringify(got) === '["squat"]', got, ["squat"]);
  got = feed(calibrated(), [{ up: 0.07 }, { up: 0.02 }, { crouch: 0.04 }, {}]);
  check("landing from a jump (a small dip) is not a squat", JSON.stringify(got) === '["jump"]', got, ["jump"]);
  const hidden = calibrated();
  const lm = body({ x: 0.3 });
  lm[23].visibility = 0.1;
  check("hips out of view: no moves rather than a guess", hidden.update(lm, 3).length === 0, "moves", "none");
}

console.log(`\n${count - failed} of ${count} checks passed${failed ? `; FAILED: ${failed}` : ""}`);
process.exit(failed ? 1 : 0);
