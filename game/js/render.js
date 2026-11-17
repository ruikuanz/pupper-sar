// Draws one player's view of the game on a canvas.
//
// The picture is built to be read by a camera as well as by a person:
//   - every obstacle kind has one saturated colour (config.json's palette), and
//     nothing else in the view is both bright and saturated;
//   - every obstacle touches the ground, so the bottom edge of its blob in the
//     image says how far away it is (the floor recedes upward);
//   - Pupper's view has a thick green frame, which is how the robot finds the
//     game on the screen and works out where it is standing.
//
// The projection is the one surfer/perception.py inverts. A point z units down
// the track, on the ground, in lane offset l (-1, 0, +1) lands at
//     s = z0 / (z + z0)
//     x = 0.5 + l * lane_width * s          (fraction of the view's width)
//     y = horizon + (player - horizon) * s  (fraction of the view's height)

import { BEAM, HURDLE, WALL } from "./engine.js";

function shade(hex, k) {
  const n = parseInt(hex.slice(1), 16);
  const c = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => Math.round(v * k));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export class ViewRenderer {
  constructor(canvas, config, role) {
    this.canvas = canvas;
    this.config = config;
    this.role = role; // "pupper" or "human"
    this.W = config.view.width;
    this.H = config.view.height;
    this.B = config.view.border;
    canvas.width = this.W + 2 * this.B;
    canvas.height = this.H + 2 * this.B;
    this.ctx = canvas.getContext("2d");
    this.shake = 0;
    this.flashUntil = 0;
    this.banner = null; // { text, until }
  }

  // Ground point (lane offset, z) plus a height above the ground -> pixels.
  project(laneOffset, z, height = 0) {
    const P = this.config.projection;
    const s = P.z0 / (z + P.z0);
    const unit = P.lane_width * this.W * s; // one lane width, in pixels, at this depth
    return {
      x: (0.5 + laneOffset * P.lane_width * s) * this.W,
      y: (P.horizon + (P.player - P.horizon) * s) * this.H - height * unit,
      unit,
      s,
    };
  }

  draw(game, player, now) {
    const { ctx, W, H, B } = this;
    const pal = this.config.palette;
    ctx.save();
    // The frame. Pupper's is the green one its camera looks for.
    ctx.fillStyle = this.role === "pupper" ? pal.border_pupper : pal.border_human;
    ctx.fillRect(0, 0, W + 2 * B, H + 2 * B);
    ctx.translate(B, B);
    ctx.beginPath();
    ctx.rect(0, 0, W, H);
    ctx.clip();
    if (this.shake > 0.3) {
      ctx.translate((Math.random() - 0.5) * this.shake, (Math.random() - 0.5) * this.shake);
      this.shake *= 0.85;
    }

    this.drawWorld(player);
    const rows = game.course.rowsBetween(player.dist - 6, player.dist + this.config.projection.far);
    // Far to near; the runner goes in between, in front of everything ahead of it.
    for (let i = rows.length - 1; i >= 0; i--) {
      if (rows[i].z - player.dist >= 0) this.drawRow(game, player, rows[i]);
    }
    this.drawRunner(game, player);
    for (let i = rows.length - 1; i >= 0; i--) {
      if (rows[i].z - player.dist < 0) this.drawRow(game, player, rows[i]);
    }
    this.drawHud(game, player, now);
    ctx.restore();
  }

  drawWorld(player) {
    const { ctx, W, H } = this;
    const P = this.config.projection;
    const pal = this.config.palette;
    const horizon = P.horizon * H;
    const sky = ctx.createLinearGradient(0, 0, 0, horizon);
    sky.addColorStop(0, pal.sky_top);
    sky.addColorStop(1, pal.sky_bottom);
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, W, horizon);
    ctx.fillStyle = pal.sky_top;
    ctx.fillRect(0, horizon, W, H - horizon);

    // The track: three lanes, from the horizon to past the bottom of the view.
    const near = this.project(-1.5, -P.z0 * 0.45);
    const nearR = this.project(1.5, -P.z0 * 0.45);
    const far = this.project(-1.5, 2000);
    const farR = this.project(1.5, 2000);
    ctx.fillStyle = pal.track;
    ctx.beginPath();
    ctx.moveTo(far.x, far.y);
    ctx.lineTo(farR.x, farR.y);
    ctx.lineTo(nearR.x, nearR.y);
    ctx.lineTo(near.x, near.y);
    ctx.fill();
    ctx.strokeStyle = pal.track_edge;
    ctx.lineWidth = 3;
    for (const [a, b] of [[far, near], [farR, nearR]]) {
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    // Lane lines as dashes fixed to the track, so they stream past with speed.
    const dash = 4;
    const offset = player.dist % (2 * dash);
    ctx.fillStyle = pal.lane_line;
    for (const lane of [-0.5, 0.5]) {
      for (let z = -offset - 2 * dash; z < P.far; z += 2 * dash) {
        if (z + dash < -P.z0 * 0.4) continue;
        const a = this.project(lane, Math.max(z, -P.z0 * 0.4));
        const b = this.project(lane, z + dash);
        const wa = 0.03 * a.unit, wb = 0.03 * b.unit;
        ctx.beginPath();
        ctx.moveTo(a.x - wa, a.y);
        ctx.lineTo(a.x + wa, a.y);
        ctx.lineTo(b.x + wb, b.y);
        ctx.lineTo(b.x - wb, b.y);
        ctx.fill();
      }
    }

    // Pylons beside the track: dark, so the camera ignores them.
    const spacing = 12;
    const first = Math.ceil((player.dist - 4) / spacing) * spacing;
    for (let z = first + 6 * spacing; z >= first; z -= spacing) {
      const rel = z - player.dist;
      for (const side of [-1, 1]) {
        const base = this.project(side * 2.1, rel);
        const w = 0.22 * base.unit;
        const h = (1.6 + ((z / spacing) % 3) * 0.5) * base.unit;
        ctx.fillStyle = pal.scenery;
        ctx.fillRect(base.x - w / 2, base.y - h, w, h);
        ctx.fillStyle = pal.track_edge;
        ctx.fillRect(base.x - w / 2, base.y - h, w, 0.12 * base.unit);
      }
    }
  }

  // A box standing on the ground: front face, and a darker top face.
  box(laneOffset, z, halfWidth, bottom, top, depth, color) {
    const { ctx } = this;
    const a = this.project(laneOffset, z);
    const b = this.project(laneOffset, z + depth);
    const x0 = a.x - halfWidth * a.unit, x1 = a.x + halfWidth * a.unit;
    const yTop = a.y - top * a.unit, yBottom = a.y - bottom * a.unit;
    ctx.fillStyle = shade(color, 0.82);
    ctx.beginPath();
    ctx.moveTo(x0, yTop);
    ctx.lineTo(x1, yTop);
    ctx.lineTo(b.x + halfWidth * b.unit, b.y - top * b.unit);
    ctx.lineTo(b.x - halfWidth * b.unit, b.y - top * b.unit);
    ctx.fill();
    ctx.fillStyle = color;
    ctx.fillRect(x0, yTop, x1 - x0, yBottom - yTop);
    return { x0, x1, yTop, yBottom, unit: a.unit };
  }

  drawRow(game, player, row) {
    const { ctx } = this;
    const pal = this.config.palette;
    const O = this.config.obstacles;
    const rel = row.z - player.dist;
    if (rel < -this.config.projection.z0 * 0.4) return;

    for (const bolt of game.course.boltsOf(row)) {
      const r = bolt.z - player.dist;
      if (r < -1 || player.collected.has(bolt.key)) continue;
      const p = this.project(bolt.lane - 1, r, 0.32);
      ctx.fillStyle = pal.bolt;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 0.11 * p.unit, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = pal.track;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 0.045 * p.unit, 0, 2 * Math.PI);
      ctx.fill();
    }

    for (let lane = 0; lane < 3; lane++) {
      const cell = row.cells[lane];
      if (cell !== HURDLE && cell !== BEAM && cell !== WALL) continue;
      const off = lane - 1;
      // An obstacle this player crashed through is knocked flat and goes grey.
      const broken = player.hit.has(`${row.i}:${lane}`);
      const color = broken ? pal.lane_line : pal[cell];
      if (broken) {
        this.box(off, rel, O[cell].half_width ?? 0.45, 0, 0.06, O[cell].depth, color);
        continue;
      }
      if (cell === WALL) {
        const f = this.box(off, rel, O.wall.half_width, 0, O.wall.height, O.wall.depth, color);
        ctx.fillStyle = shade(color, 0.7);
        for (let k = 1; k < 4; k++) ctx.fillRect(f.x0, f.yTop + (k * (f.yBottom - f.yTop)) / 4 - 0.03 * f.unit, f.x1 - f.x0, 0.06 * f.unit);
      } else if (cell === HURDLE) {
        const f = this.box(off, rel, O.hurdle.half_width, 0, O.hurdle.height, O.hurdle.depth, color);
        ctx.fillStyle = shade(color, 0.72);
        const n = 5, w = (f.x1 - f.x0) / (2 * n);
        for (let k = 0; k < n; k++) ctx.fillRect(f.x0 + (2 * k + 0.5) * w, f.yTop + 0.1 * f.unit, w, f.yBottom - f.yTop - 0.1 * f.unit);
      } else {
        // The beam: a bar overhead, on two posts, over a stripe painted on the
        // ground. The stripe is what gives the beam a bottom edge on the floor
        // in the middle of the lane, like the other obstacles have.
        const hw = O.beam.half_width;
        const a = this.project(off, rel), b = this.project(off, rel + O.beam.ground_stripe);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(a.x - hw * a.unit, a.y);
        ctx.lineTo(a.x + hw * a.unit, a.y);
        ctx.lineTo(b.x + hw * b.unit, b.y);
        ctx.lineTo(b.x - hw * b.unit, b.y);
        ctx.fill();
        this.box(off - hw + O.beam.post / 2, rel, O.beam.post / 2, 0, O.beam.top, O.beam.depth, color);
        this.box(off + hw - O.beam.post / 2, rel, O.beam.post / 2, 0, O.beam.top, O.beam.depth, color);
        this.box(off, rel, hw, O.beam.bottom, O.beam.top, O.beam.depth, color);
      }
    }
  }

  drawRunner(game, player) {
    const { ctx } = this;
    const pal = this.config.palette;
    const t = game.t;
    const base = this.project(player.x - 1, 0);
    const u = base.unit;
    const up = player.height(t) * 1.15 * u;
    const crouch = player.crouch(t);
    const running = game.state === "running" && player.alive;
    const stride = running ? Math.sin(t * 14) : 0;
    const blink = t < player.invulnerableUntil && Math.floor(t * 12) % 2 === 0;

    // Shadow: smaller while in the air.
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.beginPath();
    ctx.ellipse(base.x, base.y, (0.34 - 0.12 * player.height(t)) * u, 0.07 * u, 0, 0, 2 * Math.PI);
    ctx.fill();
    if (blink) return;

    // A robot dog seen from behind: two hind legs, body, head, ears, tail.
    const bodyH = (0.36 - 0.1 * crouch) * u;
    const legH = (0.26 - 0.2 * crouch) * u;
    const bodyW = 0.5 * u;
    const y = base.y - up;
    ctx.fillStyle = pal.dog_dark;
    for (const side of [-1, 1]) {
      const lift = Math.max(0, side * stride) * 0.1 * u;
      ctx.fillRect(base.x + side * 0.17 * u - 0.055 * u, y - legH - lift, 0.11 * u, legH);
    }
    const top = y - legH - bodyH;
    ctx.fillStyle = pal.dog;
    ctx.beginPath();
    ctx.roundRect(base.x - bodyW / 2, top, bodyW, bodyH, 0.08 * u);
    ctx.fill();
    ctx.strokeStyle = pal.dog_dark;
    ctx.lineWidth = Math.max(1.5, 0.03 * u);
    ctx.stroke();
    // Head, pulled down in a squat.
    const headY = top - (0.2 - 0.16 * crouch) * u;
    ctx.fillStyle = pal.dog;
    ctx.beginPath();
    ctx.roundRect(base.x - 0.17 * u, headY, 0.34 * u, 0.26 * u, 0.07 * u);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = pal.dog_dark;
    for (const side of [-1, 1]) {
      ctx.beginPath();
      ctx.moveTo(base.x + side * 0.17 * u, headY + 0.02 * u);
      ctx.lineTo(base.x + side * 0.1 * u, headY + 0.02 * u);
      ctx.lineTo(base.x + side * 0.15 * u, headY - 0.12 * u);
      ctx.fill();
    }
    // Tail wags with the stride.
    ctx.lineWidth = Math.max(2, 0.05 * u);
    ctx.beginPath();
    ctx.moveTo(base.x, top + 0.1 * u);
    ctx.lineTo(base.x + stride * 0.08 * u, top - 0.1 * u);
    ctx.stroke();
  }

  drawHud(game, player, now) {
    const { ctx, W, H } = this;
    const pal = this.config.palette;
    ctx.fillStyle = pal.text;
    ctx.textBaseline = "top";
    ctx.font = "700 15px ui-monospace, Menlo, monospace";
    ctx.textAlign = "left";
    ctx.fillText(this.role === "pupper" ? "PUPPER" : "HUMAN", 14, 12);
    ctx.textAlign = "right";
    ctx.fillText(String(player.score).padStart(5, "0"), W - 14, 12);
    // Lives: white pips (nothing red in the view but walls).
    for (let i = 0; i < this.config.rules.lives; i++) {
      ctx.globalAlpha = player.endless || i < player.lives ? 1 : 0.2;
      ctx.fillRect(14 + i * 16, 36, 10, 10);
    }
    ctx.globalAlpha = 1;
    ctx.textAlign = "right";
    ctx.font = "500 12px ui-monospace, Menlo, monospace";
    ctx.fillText(`bolts ${player.bolts}   crashes ${player.crashes}`, W - 14, 34);

    // The move just made, for a moment.
    if (game.t - player.lastMoveAt < 0.7 && player.lastMove !== "run" && game.state === "running") {
      ctx.textAlign = "center";
      ctx.font = "800 22px ui-monospace, Menlo, monospace";
      ctx.fillText(player.lastMove.toUpperCase(), W / 2, 62);
    }

    let big = null, small = null;
    if (game.state === "ready") {
      big = this.config.name.toUpperCase();
      small = "press START";
    } else if (game.state === "countdown") {
      big = String(Math.ceil(game.startsAt - game.t));
    } else if (!player.alive) {
      big = "GAME OVER";
      small = `score ${player.score}`;
    } else if (game.state === "over") {
      big = "STOPPED";
      small = `score ${player.score}`;
    }
    if (big) {
      ctx.fillStyle = "rgba(8,10,20,0.6)";
      ctx.fillRect(0, H * 0.34, W, H * 0.2);
      ctx.fillStyle = pal.text;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.font = "800 40px ui-monospace, Menlo, monospace";
      ctx.fillText(big, W / 2, H * 0.42);
      if (small) {
        ctx.font = "500 15px ui-monospace, Menlo, monospace";
        ctx.fillText(small, W / 2, H * 0.5);
      }
    }
    // A crash flashes white (never red: red is a wall to the camera).
    if (now < this.flashUntil) {
      ctx.fillStyle = `rgba(255,255,255,${0.5 * (this.flashUntil - now) / 0.25})`;
      ctx.fillRect(0, 0, W, H);
    }
  }

  crash(now) {
    this.shake = 14;
    this.flashUntil = now + 0.25;
  }
}
