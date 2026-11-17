// The whole loop, with no browser and no robot: the real game engine and
// renderer run here, Pupper's view goes to surfer/server.py --sim as JPEGs, and
// the moves it sends back steer the runner. It passes if Pupper finishes the
// run without a crash and has used every move.
//
//   cd tests && npm install         (once: a canvas and a WebSocket for node)
//   node tests/play_headless.mjs                 # 60 s on seed 1
//   node tests/play_headless.mjs --seconds 120 --seed 7 --save /tmp/frames
//
// It starts its own server (--port, default 8771) and stops it at the end.

import { spawn } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createCanvas } from "@napi-rs/canvas";
import WebSocket from "ws";

import { Game } from "../game/js/engine.js";
import { ViewRenderer } from "../game/js/render.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const SECONDS = Number(option("seconds", 60));
const SEED = Number(option("seed", 1));
const PRESET = option("preset", "pupper");
const SAVE = option("save", null);
const PORT = Number(option("port", 8771));
const PYTHON = option("python", join(ROOT, ".venv/bin/python"));

const config = JSON.parse(readFileSync(join(ROOT, "game/config.json"), "utf8"));
const game = new Game(config, { seed: SEED, preset: PRESET });
const { width: W, height: H, border: B, stream_width: SW, stream_height: SH, stream_fps: FPS } = config.view;
const view = new ViewRenderer(createCanvas(1, 1), config, "pupper");
const small = createCanvas(SW, SH);
const now = () => performance.now() / 1000;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const server = spawn(PYTHON, [join(ROOT, "surfer/server.py"), "--sim", "--port", String(PORT)], { stdio: ["ignore", "inherit", "inherit"] });
process.on("exit", () => server.kill());

async function connect() {
  for (let attempt = 0; attempt < 50; attempt++) {
    try {
      return await new Promise((resolve, reject) => {
        const socket = new WebSocket(`ws://127.0.0.1:${PORT}/ws`);
        socket.once("open", () => resolve(socket));
        socket.once("error", reject);
      });
    } catch {
      await sleep(200);
    }
  }
  throw new Error("the server did not start");
}

const socket = await connect();
const crashes = [];
let status = null;
let ready = null;
socket.on("message", (data, isBinary) => {
  if (isBinary) return; // the debug picture
  const message = JSON.parse(data.toString());
  if (message.type === "move") {
    const ok = game.pupper.command(message.move, game.t, { duration: message.duration });
    // For a jump or squat: when does the obstacle really arrive? Best is halfway through the move.
    const next = game.course.rowsBetween(game.pupper.dist, game.pupper.dist + 200)[0];
    const arrives = (next.z - game.pupper.dist) / game.speed;
    const timing = message.move === "jump" || message.move === "squat"
      ? `  obstacle in ${arrives.toFixed(2)} s (ideal ${(message.duration / 2).toFixed(2)})` : `  next row in ${arrives.toFixed(2)} s`;
    console.log(`  t=${game.playTime.toFixed(2)}  ${message.move}${timing}${ok ? "" : "  (refused by the game)"}`);
  } else if (message.type === "ready") {
    ready = message;
  } else if (message.type === "status") {
    status = message;
  }
});

socket.send(JSON.stringify({ type: "control", action: "start", preset: PRESET, speed: game.preset.speed }));
while (!ready) await sleep(50);
if (!ready.ok) throw new Error(`the robot can't start: ${ready.message}`);
game.pupper.endless = game.human.endless = true;
game.start();
game.pupper.endless = game.human.endless = true;

let last = now();
let lastFrame = 0, lastState = 0, saved = 0;
if (SAVE) mkdirSync(SAVE, { recursive: true });
while (game.playTime < SECONDS) {
  const t = now();
  for (const event of game.update(Math.min(0.05, t - last))) {
    if (event.player !== "pupper" || event.type !== "crash") continue;
    crashes.push(event);
    console.log(`  t=${game.playTime.toFixed(2)}  CRASH into a ${event.cell} in lane ${event.lane} (row ${event.row}); robot saw ${JSON.stringify(status?.rows)}`);
  }
  last = t;
  view.draw(game, game.pupper, t);
  if (t - lastFrame >= 1 / FPS) {
    lastFrame = t;
    small.getContext("2d").drawImage(view.canvas, B, B, W, H, 0, 0, SW, SH);
    socket.send(small.toBuffer("image/jpeg", 80));
    if (SAVE && saved < 400 && game.state === "running") {
      // The picture, and the truth about it: what is in each row ahead, and how far.
      const name = join(SAVE, `frame_${String(saved++).padStart(4, "0")}`);
      const truth = game.course.rowsBetween(game.pupper.dist - 1, game.pupper.dist + 60)
        .map((row) => ({ z: +(row.z - game.pupper.dist).toFixed(2), cells: row.cells }));
      writeFileSync(`${name}.png`, small.toBuffer("image/png"));
      writeFileSync(`${name}.json`, JSON.stringify({ lane: game.pupper.lane, rows: truth }));
    }
  }
  if (t - lastState >= 0.25) {
    lastState = t;
    socket.send(JSON.stringify({ type: "state", game: game.state, t: game.playTime }));
  }
  await sleep(1000 / 60);
}
socket.send(JSON.stringify({ type: "control", action: "stop" }));
await sleep(200);

const moves = game.pupper.moves;
const rows = game.course.rowsBetween(0, game.pupper.dist).length;
console.log(`\n${SECONDS} s on seed ${SEED} (${PRESET}): ${rows} rows, ${crashes.length} crashes, bolts ${game.pupper.bolts}`);
console.log(`moves: left ${moves.left}, right ${moves.right}, jump ${moves.jump}, squat ${moves.squat}; track speed measured ${status?.speed}`);
const unused = ["left", "right", "jump", "squat"].filter((m) => moves[m] === 0);
const failed = crashes.length > 0 || unused.length > 0;
if (unused.length) console.log(`never used: ${unused.join(", ")}`);
console.log(failed ? "FAILED" : "PASSED");
socket.close();
server.kill();
process.exit(failed ? 1 : 0);
