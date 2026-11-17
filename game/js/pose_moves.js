// Body pose -> the five moves. No DOM and no MediaPipe in this file: it takes
// the landmark list a pose model returns and says which moves the person just
// made, so it can be tested with made-up landmarks (tests/pose_test.mjs).
//
// The person plays the way Pupper does, with three places to stand:
//   step left / right   -> the lane follows where the body is (left, centre, right)
//   jump                -> hips and shoulders both rise
//   squat               -> shoulders drop
//   run in place        -> nothing to detect: it is the "do nothing" move
//
// Everything is measured against a calibration pose (standing in the centre),
// in units of the person's own shoulder width and torso length, so it does not
// matter how far they stand from the camera.
//
// Landmarks are MediaPipe Pose's: x to the right and y down, both 0..1 of the
// camera image. 11/12 are the shoulders, 23/24 the hips.

const L_SHOULDER = 11, R_SHOULDER = 12, L_HIP = 23, R_HIP = 24;

export const POSE_DEFAULTS = {
  min_visibility: 0.5,
  calibrate_s: 1.5,
  lane_enter: 0.75,   // shoulder widths off centre to step into a side lane
  lane_leave: 0.4,    // ... and to come back to the centre (smaller: hysteresis)
  jump_rise: 0.16,    // torso lengths the hips AND shoulders rise in a jump
  jump_rearm: 0.06,
  squat_drop: 0.3,    // torso lengths the shoulders drop in a squat
  squat_rearm: 0.15,
};

const median = (values) => {
  const v = [...values].sort((a, b) => a - b);
  return v[Math.floor(v.length / 2)];
};

// What we measure in one frame, or null if the body is not clearly in view.
export function measure(landmarks, minVisibility = POSE_DEFAULTS.min_visibility) {
  if (!landmarks || landmarks.length <= R_HIP) return null;
  const pts = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP].map((i) => landmarks[i]);
  if (pts.some((p) => !p || (p.visibility ?? 1) < minVisibility)) return null;
  const [ls, rs, lh, rh] = pts;
  const shoulderY = (ls.y + rs.y) / 2;
  const hipY = (lh.y + rh.y) / 2;
  return {
    // The camera image is not mirrored: a step to the person's right moves
    // them toward smaller x. Flip it, so bodyX grows to the person's right.
    bodyX: 1 - (lh.x + rh.x) / 2,
    shoulderY,
    hipY,
    shoulderWidth: Math.abs(ls.x - rs.x),
    torso: hipY - shoulderY,
  };
}

export class PoseMoves {
  constructor(options = {}) {
    this.opt = { ...POSE_DEFAULTS, ...options };
    this.base = null;      // the calibration pose
    this.samples = null;   // collected while calibrating
    this.calibrateUntil = 0;
    this.lane = 1;         // where the body is: 0 left, 1 centre, 2 right
    this.airborne = false;
    this.crouched = false;
    this.last = null;      // the last measurement, in calibrated units (for the UI)
  }

  startCalibration(now) {
    this.samples = [];
    this.calibrateUntil = now + this.opt.calibrate_s;
    this.base = null;
  }

  get calibrating() {
    return this.samples !== null;
  }

  get ready() {
    return this.base !== null;
  }

  // Feed one frame. Returns the moves made since the last frame: a list drawn
  // from "left", "right", "jump", "squat" (usually empty).
  update(landmarks, now) {
    const m = measure(landmarks, this.opt.min_visibility);
    if (!m || m.torso <= 0.02 || m.shoulderWidth <= 0.02) return [];

    if (this.samples) {
      this.samples.push(m);
      if (now >= this.calibrateUntil && this.samples.length >= 5) {
        const pick = (key) => median(this.samples.map((s) => s[key]));
        this.base = {
          bodyX: pick("bodyX"), shoulderY: pick("shoulderY"), hipY: pick("hipY"),
          shoulderWidth: pick("shoulderWidth"), torso: pick("torso"),
        };
        this.samples = null;
        this.lane = 1;
        this.airborne = this.crouched = false;
      }
      return [];
    }
    if (!this.base) return [];

    const b = this.base;
    const side = (m.bodyX - b.bodyX) / b.shoulderWidth;   // + is the person's right
    const hipRise = (b.hipY - m.hipY) / b.torso;           // + is up
    const shoulderRise = (b.shoulderY - m.shoulderY) / b.torso;
    this.last = { side, hipRise, shoulderRise };
    const moves = [];

    // Lanes: where the body is, with hysteresis so the edge of a zone doesn't flicker.
    let zone = this.lane;
    if (this.lane === 1) {
      if (side > this.opt.lane_enter) zone = 2;
      else if (side < -this.opt.lane_enter) zone = 0;
    } else if (this.lane === 2) {
      if (side < -this.opt.lane_enter) zone = 0;
      else if (side < this.opt.lane_leave) zone = 1;
    } else {
      if (side > this.opt.lane_enter) zone = 2;
      else if (side > -this.opt.lane_leave) zone = 1;
    }
    while (this.lane < zone) { this.lane += 1; moves.push("right"); }
    while (this.lane > zone) { this.lane -= 1; moves.push("left"); }

    // Jump: the whole body goes up. (Standing on tiptoe raises both too, but less.)
    const rise = Math.min(hipRise, shoulderRise);
    if (!this.airborne && rise > this.opt.jump_rise) {
      this.airborne = true;
      moves.push("jump");
    } else if (this.airborne && rise < this.opt.jump_rearm) {
      this.airborne = false;
    }

    // Squat: the shoulders come down. Landing from a jump dips them a little,
    // which is why the squat threshold is well above the jump's.
    const drop = -shoulderRise;
    if (!this.crouched && !this.airborne && drop > this.opt.squat_drop) {
      this.crouched = true;
      moves.push("squat");
    } else if (this.crouched && drop < this.opt.squat_rearm) {
      this.crouched = false;
    }
    return moves;
  }
}
