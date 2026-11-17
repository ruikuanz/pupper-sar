// The control strip for the real game (real/play.py opens it under the two
// game windows). It has no game in it: it reads the webcam and turns the
// player's steps, jumps and squats into moves, and it shows what Pupper is doing.
//
// It talks to real/play.py, not to a server: play.py gives this page one
// function, window.bridge(action, value), and calls window.panel.* back.

import { PoseCamera } from "./pose_camera.js";

const $ = (id) => document.getElementById(id);
const bridge = (action, value) => (window.bridge ? window.bridge(action, value ?? null) : Promise.resolve("no bridge"));

// ---------------------------------------------------------------- the human
const pose = new PoseCamera($("webcam"), $("skeleton"), (move) => bridge("human", move));
let wasCrouched = false;

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

setInterval(() => {
  $("pose-status").textContent = pose.status === "off" ? "webcam off" : pose.status;
  // Staying down keeps rolling: the game's roll is short, a person's squat is not.
  const crouched = pose.running && pose.moves.crouched;
  if (crouched && wasCrouched) bridge("human", "squat");
  wasCrouched = crouched;
}, 450);

// Recording: real/play.py saves the pictures and the keys of the left game, to teach Pupper.
let recording = false;
const showRecording = () => {
  $("record").textContent = recording ? "Stop recording" : "Record my play";
  $("record").classList.toggle("on", recording);
};
$("record").onclick = async () => {
  recording = !recording;
  showRecording();
  $("record-status").textContent = await bridge("record", recording);
};

// ---------------------------------------------------------------- both games
$("start-runs").onclick = async () => {
  $("runs-status").textContent = "starting";
  $("runs-status").textContent = await bridge("start_runs");
};

// Slow motion: how fast a run plays in BOTH windows (1 = the game's own speed).
const showSpeed = () => { $("speed-value").textContent = `${Number($("speed").value).toFixed(2)}\u00d7`; };
$("speed").oninput = showSpeed;
$("speed").onchange = () => bridge("speed", Number($("speed").value));

// What real/play.py was started with (--speed, --record).
bridge("settings").then((answer) => {
  try {
    const settings = JSON.parse(answer);
    $("speed").value = settings.speed;
    recording = settings.recording;
  } catch { /* opened without real/play.py: keep the defaults */ }
  showSpeed();
  showRecording();
});

// ---------------------------------------------------------------- Pupper
if (new URLSearchParams(location.search).get("pupper") === "0") $("pupper-card").style.display = "none";

let connected = false;
const buttons = ["play", "stop", "stand", "lie"];
$("play").onclick = () => bridge("start");
$("stop").onclick = () => bridge("stop");
$("stand").onclick = () => bridge("stand_up");
$("lie").onclick = () => bridge("lie_down");

window.panel = {
  // Is real/play.py connected to surfer/server.py?
  link(up) {
    connected = up;
    for (const id of buttons) $(id).disabled = !up;
    if (!up) {
      $("link-status").textContent = "not connected";
      $("robot-status").textContent = "Start surfer/server.py --game real to let Pupper play.";
    }
  },
  // A message from the server: its status 5 times a second, or the answer to "Let Pupper play".
  message(m) {
    if (!connected) return;
    if (m.type === "ready") {
      if (!m.ok) $("robot-status").textContent = `Pupper can't start: ${m.message}`;
      return;
    }
    if (m.type !== "status") return;
    const lanes = ["left", "centre", "right"];
    $("link-status").textContent = `${m.mode || "robot"}: ${m.state}`;
    const run = m.run ? `in a run (${m.eyes} eyes), runner in the ${lanes[m.game_lane]} lane` : "no run on screen";
    const dog = `dog: ${m.move}, ${lanes[m.lane]} spot${m.behind ? `, ${m.behind} move${m.behind > 1 ? "s" : ""} behind` : ""}`;
    $("robot-status").textContent = `${run}. ${dog}. ${m.fps} pictures/s${m.message ? `. Last stop: ${m.message}` : ""}`;
  },
  // How the recording is going (once a second).
  recorder(text) {
    $("record-status").textContent = text === "not recording" ? "" : text;
  },
  // What Pupper sees (a JPEG, base64).
  eye(jpeg) {
    $("pupper-eye").src = `data:image/jpeg;base64,${jpeg}`;
  },
};
