// The page: one Game, two views of it, and the three ways moves come in
// (the robot link, the webcam, the keyboard).

import { Game } from "./engine.js";
import { ViewRenderer } from "./render.js";
import { RobotLink } from "./link.js";
import { PoseCamera } from "./pose_camera.js";

const $ = (id) => document.getElementById(id);
const config = await (await fetch("config.json")).json();

const game = new Game(config, { seed: 1, preset: "pupper" });
const views = {
  pupper: new ViewRenderer($("pupper-view"), config, "pupper"),
  human: new ViewRenderer($("human-view"), config, "human"),
};

for (const [key, preset] of Object.entries(config.presets)) {
  $("preset").add(new Option(preset.label, key));
}

// ---------------------------------------------------------------- the robot
let robot = { state: "", ready: false, mode: "" };  // the last status it sent
let waitingForRobot = null;                          // timer while "start" waits for Pupper to stand

const link = new RobotLink({
  onOpen: () => showLink(),
  onClose: () => {
    robot = { state: "", ready: false, mode: "" };
    showLink();
  },
  // A move Pupper has just started. Its duration is how long the move takes
  // the real body, so the runner on screen moves as fast as the dog does.
  onMove: (m) => game.pupper.command(m.move, game.t, { duration: m.duration }),
  onReady: (m) => {
    if (!waitingForRobot) return;
    clearTimeout(waitingForRobot);
    waitingForRobot = null;
    if (m.ok) game.start();
    else $("robot-status").textContent = `Pupper can't start: ${m.message}`;
  },
  onStatus: (s) => {
    robot = s;
    showLink();
  },
  onDebugImage: (blob) => {
    const img = $("pupper-eye");
    if (img.src) URL.revokeObjectURL(img.src);
    img.src = URL.createObjectURL(blob);
  },
});

function showLink() {
  if (!link.connected) {
    $("link-status").textContent = "not connected";
    $("robot-status").textContent = "Start surfer/server.py to let Pupper play. Keys A D W S play this view by hand.";
    return;
  }
  const lanes = ["left", "centre", "right"];
  $("link-status").textContent = `${robot.mode || "robot"}: ${robot.state || "connecting"}`;
  const bits = [];
  if (robot.lane !== undefined) bits.push(`lane ${lanes[robot.lane]}`);
  if (robot.source) bits.push(`eyes: ${robot.source}${robot.screen === false ? " (screen not found)" : ""}`);
  if (robot.fps) bits.push(`${robot.fps.toFixed(0)} fps`);
  if (robot.position) bits.push(`${(100 * robot.position.left).toFixed(0)} cm left, ${(100 * robot.position.distance).toFixed(0)} cm away`);
  if (robot.message) bits.push(robot.message);
  $("robot-status").textContent = bits.join(" · ");
}

// Pupper's view as its vision gets it when it is not looking through its
// camera: the inside of the green frame, shrunk, as a JPEG.
const streamCanvas = document.createElement("canvas");
streamCanvas.width = config.view.stream_width;
streamCanvas.height = config.view.stream_height;
const streamCtx = streamCanvas.getContext("2d");
let lastFrameAt = 0;

function streamFrame(now) {
  if (!link.connected || robot.source !== "stream") return;
  if (now - lastFrameAt < 1 / config.view.stream_fps) return;
  lastFrameAt = now;
  const { width: W, height: H, border: B } = config.view;
  streamCtx.drawImage($("pupper-view"), B, B, W, H, 0, 0, streamCanvas.width, streamCanvas.height);
  streamCanvas.toBlob((blob) => blob && link.sendFrame(blob), "image/jpeg", 0.8);
}

// ---------------------------------------------------------------- the human
const pose = new PoseCamera($("webcam"), $("skeleton"), (move) => game.human.command(move, game.t));

$("camera").onclick = async () => {
  if (pose.running) {
    pose.stop();
    $("camera").textContent = "Use webcam";
    $("calibrate").disabled = true;
    return;
  }
  try {
    $("camera").disabled = true;
    await pose.start();
    $("camera").textContent = "Webcam off";
    $("calibrate").disabled = false;
  } catch (error) {
    pose.status = `webcam failed: ${error.message}`;
  } finally {
    $("camera").disabled = false;
  }
};
$("calibrate").onclick = () => pose.calibrate();

// ---------------------------------------------------------------- controls
function start() {
  const seed = Math.max(1, parseInt($("seed").value, 10) || 1);
  game.newCourse(seed, $("preset").value);
  game.pupper.endless = game.human.endless = $("endless").checked;
  game.state = "ready";
  clearTimeout(waitingForRobot);
  waitingForRobot = null;
  if (!link.connected) {
    game.start();
    return;
  }
  // Pupper has to be standing before the countdown: ask, and wait for "ready".
  $("robot-status").textContent = "waking Pupper…";
  link.send({ type: "control", action: "start", preset: $("preset").value, speed: game.preset.speed });
  waitingForRobot = setTimeout(() => {
    waitingForRobot = null;
    $("robot-status").textContent = "Pupper did not answer; starting without it.";
    game.start();
  }, 15000);
}

function stop() {
  clearTimeout(waitingForRobot);
  waitingForRobot = null;
  game.stop();
  link.send({ type: "control", action: "stop" });
}

$("start").onclick = start;
$("stop").onclick = stop;
$("stand").onclick = () => link.send({ type: "control", action: "stand_up" });
$("lie").onclick = () => link.send({ type: "control", action: "lie_down" });
$("endless").onchange = () => (game.pupper.endless = game.human.endless = $("endless").checked);
$("focus").onchange = () => document.body.classList.toggle("focus-pupper", $("focus").checked);

const HUMAN_KEYS = { ArrowLeft: "left", ArrowRight: "right", ArrowUp: "jump", ArrowDown: "squat" };
const PUPPER_KEYS = { KeyA: "left", KeyD: "right", KeyW: "jump", KeyS: "squat" };

window.addEventListener("keydown", (event) => {
  if (event.target.matches("input, select")) return;
  if (event.code === "Space") {
    event.preventDefault();
    start();
  } else if (event.code === "Escape") {
    stop();
  } else if (event.code === "KeyF") {
    $("focus").checked = !$("focus").checked;
    $("focus").onchange();
  } else if (HUMAN_KEYS[event.code]) {
    event.preventDefault();
    if (!event.repeat) game.human.command(HUMAN_KEYS[event.code], game.t);
  } else if (PUPPER_KEYS[event.code] && !event.repeat) {
    // By hand, for trying the game without a robot. Ignored while Pupper plays.
    if (!link.connected || robot.state !== "playing") game.pupper.command(PUPPER_KEYS[event.code], game.t);
  }
});

// ---------------------------------------------------------------- the loop
let last = performance.now() / 1000;
let lastStateAt = 0;
let pupperWasAlive = true;

function frame() {
  const now = performance.now() / 1000;
  const dt = Math.min(0.05, now - last);
  last = now;

  // A person who stays crouched keeps squatting.
  if (pose.running && pose.moves.crouched && !game.human.squat) game.human.command("squat", game.t);

  for (const event of game.update(dt)) {
    if (event.type === "crash") views[event.player].crash(now);
  }
  views.pupper.draw(game, game.pupper, now);
  views.human.draw(game, game.human, now);
  streamFrame(now);

  // Tell the robot when its run is over, and keep it posted (it stands still
  // if this page goes quiet).
  const pupperAlive = game.pupper.alive && (game.state === "running" || game.state === "countdown");
  if (pupperWasAlive && !pupperAlive && game.state !== "ready") link.send({ type: "control", action: "stop" });
  pupperWasAlive = pupperAlive;
  if (now - lastStateAt > 0.25) {
    lastStateAt = now;
    link.send({
      type: "state", game: game.state, alive: game.pupper.alive, t: game.playTime, speed: game.speed,
      lane: game.pupper.lane, score: game.pupper.score, crashes: game.pupper.crashes,
    });
    $("pose-status").textContent = pose.status === "off" ? "webcam off" : pose.status;
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// For tests and for poking around in the console.
window.laneHound = { game, views, link, pose, config };
