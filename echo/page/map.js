// The map page: draws what echo/server.py reports, and sends it the buttons.
//
// Everything arrives in the room frame (metres; the origin is where Pupper
// started, x the way its nose pointed then, y to its left). On screen, x is
// drawn UP and y to the LEFT, so the map is seen from behind Pupper's starting
// pose: its "ahead" is up, its left is your left.

const $ = (id) => document.getElementById(id);
const mapCanvas = $("map");
const profileCanvas = $("profiles");
let report = null;        // the latest message from the server
let view = { cx: 0, cy: 0, scale: 80 };  // room point at the canvas centre, and pixels per metre
let socket = null;

// ---------------------------------------------------------------- the link
function connect() {
  socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  socket.onopen = () => { $("link").textContent = "connected"; $("link").classList.remove("off"); };
  socket.onmessage = (event) => { report = JSON.parse(event.data); show(); };
  socket.onclose = () => {
    $("link").textContent = "not connected"; $("link").classList.add("off");
    setTimeout(connect, 1000);
  };
}
function send(message) {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}
// The server stops Pupper if it does not hear from the page: say hello twice a second.
setInterval(() => send({ do: "hello" }), 500);

for (const id of ["stand", "look", "explore", "stop", "lie", "forget"]) {
  $(id).addEventListener("click", () => send({ do: id }));
}
$("walk_test").addEventListener("click", () => {
  if (confirm("Pupper will walk 1 m straight ahead WITHOUT listening. Is the floor clear?")) send({ do: "walk_test", m: 1.0 });
});
window.addEventListener("keydown", (event) => { if (event.key === " " || event.key === "Escape") send({ do: "stop" }); });

// ---------------------------------------------------------------- colors (named in style.css)
function color(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
}

// ---------------------------------------------------------------- room <-> screen
function fit(canvas) {
  // Match the canvas's pixels to its size on the page (sharp on any screen).
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const g = canvas.getContext("2d");
  g.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { g, width, height };
}
function toScreen(p, width, height) {
  return [width / 2 - (p[1] - view.cy) * view.scale, height / 2 - (p[0] - view.cx) * view.scale];
}
function toRoom(sx, sy, width, height) {
  return [view.cx - (sy - height / 2) / view.scale, view.cy - (sx - width / 2) / view.scale];
}
function frame(width, height) {
  // Choose the view so everything known fits, with a margin. It only ever
  // changes smoothly, so the map does not jump about while Pupper walks.
  const points = [[-1.5, -1.5], [1.5, 1.5], [report.pose.x, report.pose.y]];
  for (const w of report.map.walls) points.push(w.a, w.b);
  for (const c of report.map.corners) points.push(c.at);
  for (const p of report.outline) if (Math.hypot(p[0] - report.pose.x, p[1] - report.pose.y) < 7) points.push(p);
  if (report.truth) for (const w of report.truth.walls) points.push(w[0], w[1]);
  if (report.goal) points.push(report.goal);
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  const x0 = Math.min(...xs) - 0.4, x1 = Math.max(...xs) + 0.4, y0 = Math.min(...ys) - 0.4, y1 = Math.max(...ys) + 0.4;
  const want = { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, scale: Math.min(width / (y1 - y0), height / (x1 - x0)) };
  for (const key of ["cx", "cy", "scale"]) view[key] += 0.15 * (want[key] - view[key]);
}

// ---------------------------------------------------------------- the map
function drawMap() {
  const { g, width, height } = fit(mapCanvas);
  frame(width, height);
  const S = (p) => toScreen(p, width, height);
  const path = (points, close) => {
    g.beginPath();
    points.forEach((p, i) => { const [x, y] = S(p); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
    if (close) g.closePath();
  };
  g.clearRect(0, 0, width, height);
  g.lineCap = "round";
  g.lineJoin = "round";

  // The room, if every confirmed wall ran on to meet its neighbours.
  const near = report.outline.every((p) => Math.hypot(p[0] - report.pose.x, p[1] - report.pose.y) < 20);
  if (report.outline.length >= 3 && near) {
    path(report.outline, true);
    g.fillStyle = color("room");
    g.fill();
    g.strokeStyle = color("maybe");
    g.lineWidth = 1;
    g.stroke();
  }

  // A grid line every half metre, a label every metre.
  const [top, left] = toRoom(0, 0, width, height), [bottom, right] = toRoom(width, height, width, height);
  g.strokeStyle = color("grid");
  g.fillStyle = color("soft");
  g.font = "11px system-ui, sans-serif";
  g.lineWidth = 1;
  for (let x = Math.ceil(bottom * 2) / 2; x <= top; x += 0.5) {
    const y = S([x, 0])[1];
    g.beginPath(); g.moveTo(0, y); g.lineTo(width, y); g.stroke();
    if (Number.isInteger(x)) g.fillText(`${x} m`, 4, y - 3);
  }
  for (let y = Math.ceil(right * 2) / 2; y <= left; y += 0.5) {
    const x = S([0, y])[0];
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, height); g.stroke();
  }

  // Simulation only: the real room and the real dog, faintly.
  if (report.truth) {
    g.strokeStyle = color("truth");
    g.setLineDash([5, 5]);
    for (const w of report.truth.walls) { path(w); g.stroke(); }
    for (const post of report.truth.posts) {
      const [x, y] = S(post.at);
      g.beginPath(); g.arc(x, y, Math.max(3, post.radius * view.scale), 0, 2 * Math.PI); g.stroke();
    }
    g.setLineDash([]);
    drawDog(g, S, report.truth.pose.at, report.truth.pose.heading_deg, color("truth"), false);
  }

  // Walls: solid once heard from two spots, dashed while heard only once.
  for (const w of report.map.walls) {
    g.strokeStyle = color(w.confirmed ? "wall" : "maybe");
    g.lineWidth = w.confirmed ? 4 : 2;
    g.setLineDash(w.confirmed ? [] : [6, 6]);
    path([w.a, w.b]); g.stroke();
  }
  g.setLineDash([]);
  for (const c of report.map.corners) {
    const [x, y] = S(c.at);
    g.fillStyle = color(c.confirmed ? "corner" : "maybe");
    g.beginPath(); g.arc(x, y, 5, 0, 2 * Math.PI); g.fill();
  }

  // Where Pupper has been.
  if (report.trail.length > 1) {
    g.strokeStyle = color("trail");
    g.globalAlpha = 0.5;
    g.lineWidth = 2;
    path(report.trail); g.stroke();
    g.globalAlpha = 1;
  }
  g.fillStyle = color("trail");
  for (const p of report.map.spots) { const [x, y] = S(p); g.beginPath(); g.arc(x, y, 3, 0, 2 * Math.PI); g.fill(); }

  // This look's echoes: a line from Pupper to each wall's nearest point, and the face-on patch there.
  const here = [report.pose.x, report.pose.y];
  g.strokeStyle = color("echo");
  for (const e of report.echoes) {
    const a = e.bearing_deg * Math.PI / 180;
    const foot = [here[0] + e.distance_m * Math.cos(a), here[1] + e.distance_m * Math.sin(a)];
    const half = 0.22;
    g.globalAlpha = 0.35 + 0.65 * Math.min(1, e.strength);
    g.lineWidth = 1;
    g.setLineDash([2, 4]);
    path([here, foot]); g.stroke();
    g.setLineDash([]);
    g.lineWidth = 3;
    path([[foot[0] + half * Math.sin(a), foot[1] - half * Math.cos(a)], [foot[0] - half * Math.sin(a), foot[1] + half * Math.cos(a)]]);
    g.stroke();
  }
  g.globalAlpha = 1;

  // The goal, and Pupper.
  if (report.goal) {
    const [x, y] = S(report.goal);
    g.strokeStyle = color("dog");
    g.lineWidth = 2;
    g.beginPath(); g.arc(x, y, 9, 0, 2 * Math.PI); g.moveTo(x - 13, y); g.lineTo(x + 13, y); g.moveTo(x, y - 13); g.lineTo(x, y + 13); g.stroke();
  }
  drawDog(g, S, here, report.pose.heading_deg, color("dog"), true);
}

function drawDog(g, S, at, headingDeg, fill, solid) {
  // A body 0.36 m long and 0.2 m wide, with a nose.
  const a = headingDeg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
  const body = [[0.18, 0.1], [0.26, 0], [0.18, -0.1], [-0.18, -0.1], [-0.18, 0.1]];
  g.beginPath();
  body.forEach(([bx, by], i) => {
    const [x, y] = S([at[0] + c * bx - s * by, at[1] + s * bx + c * by]);
    if (i) g.lineTo(x, y); else g.moveTo(x, y);
  });
  g.closePath();
  g.lineWidth = 2;
  g.strokeStyle = fill;
  g.fillStyle = fill;
  if (solid) g.fill(); else g.stroke();
}

mapCanvas.addEventListener("click", (event) => {
  if (!report) return;
  const box = mapCanvas.getBoundingClientRect();
  const [x, y] = toRoom(event.clientX - box.left, event.clientY - box.top, box.width, box.height);
  send({ do: "goto", x, y });
});

// ---------------------------------------------------------------- what each ear heard
const SHOW_TO_M = 3.5;   // wall distance at the right edge of the plot
const SHOW_TO = 1.2;     // strength at the top (1 = a perfect mirror)
function drawProfiles() {
  const { g, width, height } = fit(profileCanvas);
  g.clearRect(0, 0, width, height);
  const pad = { left: 26, bottom: 16, top: 4 };
  const X = (m) => pad.left + (width - pad.left - 4) * m / SHOW_TO_M;
  const Y = (v) => pad.top + (height - pad.top - pad.bottom) * (1 - Math.min(v, SHOW_TO) / SHOW_TO);
  g.font = "10px system-ui, sans-serif";
  g.fillStyle = color("soft");
  g.strokeStyle = color("grid");
  g.lineWidth = 1;
  for (let m = 0; m <= SHOW_TO_M; m += 0.5) {
    g.beginPath(); g.moveTo(X(m), pad.top); g.lineTo(X(m), height - pad.bottom); g.stroke();
    if (Number.isInteger(m)) g.fillText(`${m} m`, X(m) - 8, height - 3);
  }
  for (const v of [0, 0.5, 1]) {
    g.beginPath(); g.moveTo(pad.left, Y(v)); g.lineTo(width - 4, Y(v)); g.stroke();
    g.fillText(v.toFixed(1), 2, Y(v) + 3);
  }
  if (!report || !report.profiles) return;
  const wall = report.profiles.wall_m;
  report.profiles.ears.forEach((curve, ear) => {
    g.strokeStyle = color(`ear-${ear + 1}`);
    g.lineWidth = 1.5;
    g.beginPath();
    let started = false;
    for (let i = 0; i < curve.length; i++) {
      if (wall[i] < 0.05 || wall[i] > SHOW_TO_M) continue;
      if (started) g.lineTo(X(wall[i]), Y(curve[i])); else { g.moveTo(X(wall[i]), Y(curve[i])); started = true; }
    }
    g.stroke();
  });
}

// ---------------------------------------------------------------- the words
function show() {
  const names = report.ears.names;
  $("ears").textContent = `${report.ears.mode} ears${report.ears.needs_turning ? " · turns to listen" : " · one ping"}`;
  $("state").textContent = `${report.robot.mode} · ${report.robot.problem || "standing"} · ${report.state.toLowerCase()}`;
  $("message").textContent = report.message;
  $("truth-key").hidden = !report.truth;
  $("problems").replaceChildren(...report.problems.map((line) => Object.assign(document.createElement("li"), { textContent: line })));
  $("ear-key").replaceChildren(...names.map((name, ear) => {
    const item = document.createElement("li");
    const swatch = document.createElement("i");
    swatch.className = "swatch";
    swatch.style.borderTopColor = color(`ear-${ear + 1}`);
    item.append(swatch, name.replace("_", " "));
    return item;
  }));
  const rows = [...report.echoes].sort((a, b) => a.distance_m - b.distance_m).map((e) => {
    const row = document.createElement("tr");
    const off = ((e.bearing_deg - report.pose.heading_deg + 540) % 360) - 180;   // relative to the nose
    const where = Math.abs(off) < 20 ? "ahead" : Math.abs(off) > 160 ? "behind" : off > 0 ? "left" : "right";
    const cells = [`${e.distance_m.toFixed(2)} m`, `${where} (${off > 0 ? "+" : ""}${off.toFixed(0)}°)`, ""];
    cells.forEach((text) => row.append(Object.assign(document.createElement("td"), { textContent: text })));
    const bar = Object.assign(document.createElement("span"), { className: "bar" });
    bar.style.width = `${Math.round(48 * Math.min(1, e.strength))}px`;
    row.lastChild.append(bar, e.strength.toFixed(2));
    return row;
  });
  $("echoes").tBodies[0].replaceChildren(...rows);
  const logBox = $("log");
  const text = report.log.join("\n");
  if (logBox.textContent !== text) { logBox.textContent = text; logBox.scrollTop = logBox.scrollHeight; }
  drawMap();
  drawProfiles();
}

window.addEventListener("resize", () => { if (report) show(); });
connect();
