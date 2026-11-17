// The game itself: a deterministic three-lane runner. No DOM in this file, so
// the same code runs in the browser (two views of one course) and under node
// (tests/engine_test.mjs).
//
//   Course   the obstacle rows, generated from a seed. Both players run the
//            same Course object, so the two views are exact replicas.
//   Player   one runner: lane, jump, squat, lives, score.
//   Game     the clock, the speed, and the two players.
//
// The five moves, and what they are on the phone and on Pupper:
//   run    do nothing            running in place
//   left   swipe left            shuffle left
//   right  swipe right           shuffle right
//   squat  swipe down            squat
//   jump   swipe up              jump
//
// Distances are in "track units" (a lane is about one unit wide); z is how far
// down the track something is. Lanes are 0 (left), 1 (centre), 2 (right).

export const FREE = "free";
export const HURDLE = "hurdle"; // low: jump it
export const BEAM = "beam";     // overhead: squat under it
export const WALL = "wall";     // full height: be in another lane
export const MOVES = ["run", "left", "right", "squat", "jump"];

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// The first rows are scripted so a short demo shows every move: a jump, a
// squat, then two lane changes. After that the rows are random.
const INTRO = [
  [HURDLE, HURDLE, HURDLE],
  [BEAM, BEAM, BEAM],
  [FREE, WALL, FREE],
  [WALL, FREE, WALL],
  [HURDLE, FREE, BEAM],
];

// A runner changes at most one lane between two rows (Pupper's shuffle takes
// most of a second). So a row is fair only if every lane that was passable in
// the row before has a passable lane next to it (or itself) in this one.
export function reachable(prev, cells) {
  for (let p = 0; p < 3; p++) {
    if (prev[p] === WALL) continue;
    const ok = [p - 1, p, p + 1].some((q) => q >= 0 && q < 3 && cells[q] !== WALL);
    if (!ok) return false;
  }
  return true;
}

export class Course {
  constructor(seed, preset) {
    this.seed = seed;
    this.rand = mulberry32(seed);
    this.rows = [];
    this.gap = preset.row_gap_s * preset.speed;   // distance between rows
    this.first = preset.warmup_s * preset.speed;  // distance to the first row
  }

  row(i) {
    while (this.rows.length <= i) this.rows.push(this._next());
    return this.rows[i];
  }

  // Every row with zMin < z <= zMax.
  rowsBetween(zMin, zMax) {
    const out = [];
    const start = Math.max(0, Math.floor((zMin - this.first) / this.gap) + 1);
    for (let i = start; this.first + i * this.gap <= zMax; i++) out.push(this.row(i));
    return out;
  }

  _pick(list) {
    return list[Math.floor(this.rand() * list.length)];
  }

  _pattern() {
    const r = this.rand();
    const cells = [FREE, FREE, FREE];
    const lanes = [0, 1, 2];
    if (r < 0.2) {
      cells[this._pick(lanes)] = HURDLE;
    } else if (r < 0.35) {
      cells[this._pick(lanes)] = BEAM;
    } else if (r < 0.5) {
      cells[this._pick(lanes)] = WALL;
    } else if (r < 0.65) {
      const open = this._pick(lanes);
      for (const l of lanes) if (l !== open) cells[l] = WALL;
    } else if (r < 0.77) {
      cells.fill(HURDLE);
    } else if (r < 0.89) {
      cells.fill(BEAM);
    } else {
      const kinds = [WALL, HURDLE, BEAM];
      for (const l of lanes) cells[l] = kinds.splice(Math.floor(this.rand() * kinds.length), 1)[0];
    }
    return cells;
  }

  _next() {
    const i = this.rows.length;
    const prev = i ? this.rows[i - 1].cells : [FREE, FREE, FREE];
    let cells = i < INTRO.length ? INTRO[i].slice() : null;
    for (let tries = 0; !cells && tries < 50; tries++) {
      const candidate = this._pattern();
      if (reachable(prev, candidate)) cells = candidate;
    }
    if (!cells) cells = [FREE, FREE, FREE];
    // A short line of bolts before the row, in a lane you can pass it in.
    const open = [0, 1, 2].filter((l) => cells[l] !== WALL);
    const bolts = this.rand() < 0.75 ? open[Math.floor(this.rand() * open.length)] : null;
    return { i, z: this.first + i * this.gap, cells, bolts };
  }

  // Where the bolts of a row are: three in a line, half a gap before it.
  boltsOf(row) {
    if (row.bolts === null) return [];
    const mid = row.z - this.gap / 2;
    return [-1, 0, 1].map((k) => ({ key: `${row.i}:${k}`, lane: row.bolts, z: mid + k * 2.5 }));
  }
}

const clamp01 = (v) => Math.max(0, Math.min(1, v));
const ease = (p) => p * p * (3 - 2 * p);

export class Player {
  constructor(name, course, config) {
    this.name = name;
    this.course = course;
    this.timing = config.timing;
    this.rules = config.rules;
    this.endless = false;
    this.reset();
  }

  reset() {
    this.lane = 1;        // the lane it is in, or heading for
    this.x = 1;           // where it really is, 0..2 (moves smoothly between lanes)
    this.lateral = null;  // { from, to, t0, dur }
    this.jump = null;     // { t0, dur }
    this.squat = null;    // { t0, dur }
    this.dist = 0;
    this.lives = this.rules.lives;
    this.bolts = 0;
    this.crashes = 0;
    this.alive = true;
    this.invulnerableUntil = 0;
    this.lastMove = "run";
    this.lastMoveAt = -10;
    this.hit = new Set();       // "row:lane" of obstacles it crashed through
    this.collected = new Set(); // bolt keys
    this.moves = { run: 0, left: 0, right: 0, squat: 0, jump: 0 };
  }

  get score() {
    return Math.floor(this.dist) + this.rules.bolt_points * this.bolts;
  }

  // One of the five moves. opts.duration: how long the move takes; the robot
  // sends its own (a shuffle takes Pupper most of a second), a person uses the
  // game's defaults. Returns whether the move was accepted.
  command(move, t, opts = {}) {
    if (!this.alive) return false;
    if (move === "run") return true;
    if (move === "left" || move === "right") {
      const to = this.lane + (move === "left" ? -1 : 1);
      if (to < 0 || to > 2) return false;
      this.lateral = { from: this.x, to, t0: t, dur: opts.duration ?? this.timing.lane_change_s };
      this.lane = to;
    } else if (move === "jump") {
      if (this.jump || this.squat) return false;
      this.jump = { t0: t, dur: opts.duration ?? this.timing.jump_s };
    } else if (move === "squat") {
      if (this.jump || this.squat) return false;
      this.squat = { t0: t, dur: opts.duration ?? this.timing.squat_s };
    } else {
      return false;
    }
    this.lastMove = move;
    this.lastMoveAt = t;
    this.moves[move] += 1;
    return true;
  }

  // 0 on the ground, 1 at the top of a jump.
  height(t) {
    if (!this.jump) return 0;
    return Math.sin(Math.PI * clamp01((t - this.jump.t0) / this.jump.dur));
  }

  // 0 standing, 1 fully down.
  crouch(t) {
    if (!this.squat) return 0;
    const p = clamp01((t - this.squat.t0) / this.squat.dur);
    return clamp01(Math.min(p, 1 - p) / 0.15);
  }

  occupiedLane() {
    return Math.max(0, Math.min(2, Math.round(this.x)));
  }

  update(t, dt, speed) {
    if (this.lateral) {
      const p = clamp01((t - this.lateral.t0) / this.lateral.dur);
      this.x = this.lateral.from + (this.lateral.to - this.lateral.from) * ease(p);
      if (p >= 1) this.lateral = null;
    }
    if (this.jump && t >= this.jump.t0 + this.jump.dur) this.jump = null;
    if (this.squat && t >= this.squat.t0 + this.squat.dur) this.squat = null;
    if (!this.alive) return [];

    const before = this.dist;
    this.dist += speed * dt;
    const events = [];
    const lane = this.occupiedLane();

    // Bolts and rows whose line the runner crossed in this step.
    for (const row of this.course.rowsBetween(before - this.course.gap, this.dist + this.course.gap)) {
      for (const bolt of this.course.boltsOf(row)) {
        if (bolt.z > before && bolt.z <= this.dist && bolt.lane === lane && !this.collected.has(bolt.key)) {
          this.collected.add(bolt.key);
          this.bolts += 1;
          events.push({ type: "bolt" });
        }
      }
      if (row.z > before && row.z <= this.dist) {
        const cell = row.cells[lane];
        const cleared =
          cell === FREE ||
          (cell === HURDLE && this.height(t) >= this.rules.jump_clear_height) ||
          (cell === BEAM && this.crouch(t) >= this.rules.squat_clear_depth);
        if (cleared) {
          events.push({ type: "pass", row: row.i, cell });
        } else if (t >= this.invulnerableUntil) {
          this.hit.add(`${row.i}:${lane}`);
          this.crashes += 1;
          this.invulnerableUntil = t + this.timing.invulnerable_s;
          if (!this.endless) this.lives -= 1;
          if (this.lives <= 0) this.alive = false;
          events.push({ type: "crash", row: row.i, cell, lane });
        }
      }
    }
    return events;
  }
}

export class Game {
  constructor(config, { seed = 1, preset = "pupper" } = {}) {
    this.config = config;
    this.newCourse(seed, preset);
    this.state = "ready"; // ready -> countdown -> running -> over
    this.t = 0;           // game clock, seconds; runs during countdown and play
    this.startsAt = 0;
  }

  newCourse(seed, presetName) {
    this.seed = seed;
    this.presetName = presetName;
    this.preset = this.config.presets[presetName];
    this.course = new Course(seed, this.preset);
    const endless = this.players ? this.players.map((p) => p.endless) : [false, false];
    this.pupper = new Player("pupper", this.course, this.config);
    this.human = new Player("human", this.course, this.config);
    this.players = [this.pupper, this.human];
    this.players.forEach((p, i) => (p.endless = endless[i]));
  }

  start() {
    this.newCourse(this.seed, this.presetName);
    this.t = 0;
    this.startsAt = this.config.timing.countdown_s;
    this.state = "countdown";
  }

  stop() {
    if (this.state !== "ready") this.state = "over";
  }

  // Seconds of play so far (0 during the countdown).
  get playTime() {
    return Math.max(0, this.t - this.startsAt);
  }

  get speed() {
    const p = this.preset;
    return Math.min(p.max_speed, p.speed + p.ramp * this.playTime);
  }

  // Advance the clock. Returns [{ player, type, ... }] for crashes, bolts, passes.
  update(dt) {
    if (this.state !== "countdown" && this.state !== "running") return [];
    this.t += dt;
    if (this.state === "countdown") {
      if (this.t < this.startsAt) return [];
      this.state = "running";
      dt = this.t - this.startsAt;
    }
    const events = [];
    for (const player of this.players) {
      for (const e of player.update(this.t, dt, this.speed)) events.push({ player: player.name, ...e });
    }
    if (this.players.every((p) => !p.alive)) this.state = "over";
    return events;
  }
}
