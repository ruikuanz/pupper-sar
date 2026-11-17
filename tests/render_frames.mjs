// Render Pupper's view of a course at chosen moments, with the truth next to
// each picture, for replaying through perception offline (tests/replay.py).
//
//   node tests/render_frames.mjs --seed 5 --from 40 --to 50 --out /tmp/frames
//
// The runner stays in the lane given by --lane (default: centre).

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createCanvas } from "@napi-rs/canvas";
import { Game } from "../game/js/engine.js";
import { ViewRenderer } from "../game/js/render.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const option = (name, fallback) => (args.includes(`--${name}`) ? args[args.indexOf(`--${name}`) + 1] : fallback);
const config = JSON.parse(readFileSync(join(ROOT, "game/config.json"), "utf8"));
const game = new Game(config, { seed: Number(option("seed", 1)), preset: option("preset", "pupper") });
const from = Number(option("from", 0)), to = Number(option("to", 10)), fps = Number(option("fps", 15));
const out = option("out", "frames");
mkdirSync(out, { recursive: true });

const { width: W, height: H, border: B, stream_width: SW, stream_height: SH } = config.view;
const view = new ViewRenderer(createCanvas(1, 1), config, "pupper");
const small = createCanvas(SW, SH);
game.state = "running";
game.pupper.endless = true;
game.pupper.lane = game.pupper.x = Number(option("lane", 1));
let n = 0;
for (let t = from; t <= to; t += 1 / fps) {
  game.t = t;
  game.pupper.dist = game.preset.speed * t;
  view.draw(game, game.pupper, t);
  small.getContext("2d").drawImage(view.canvas, B, B, W, H, 0, 0, SW, SH);
  const name = join(out, `frame_${String(n++).padStart(4, "0")}`);
  const rows = game.course.rowsBetween(game.pupper.dist - 1, game.pupper.dist + 70)
    .map((row) => ({ z: +(row.z - game.pupper.dist).toFixed(2), cells: row.cells }));
  writeFileSync(`${name}.png`, small.toBuffer("image/png"));
  writeFileSync(`${name}.json`, JSON.stringify({ t, lane: game.pupper.lane, rows }));
}
console.log(`${n} frames in ${out}`);
