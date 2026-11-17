// The laptop webcam and MediaPipe Pose, feeding pose_moves.js.
//
// The pose model runs in the browser (WebAssembly + WebGL), about 30 times a
// second. It is loaded from a CDN the first time the camera is switched on, so
// that first start needs internet; the game itself does not.
//
// Browsers only hand out the camera on a "secure" page: http://localhost counts
// (that is what the SSH tunnel gives you), http://pupper.local does not.

import { PoseMoves } from "./pose_moves.js";

const VISION = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

// Lines of the skeleton drawn over the preview (pairs of landmark numbers).
const BONES = [
  [11, 12], [11, 23], [12, 24], [23, 24], [11, 13], [13, 15], [12, 14], [14, 16],
  [23, 25], [25, 27], [24, 26], [26, 28],
];

export class PoseCamera {
  constructor(video, overlay, onMove) {
    this.video = video;
    this.overlay = overlay;
    this.onMove = onMove;       // called with "left" | "right" | "jump" | "squat"
    this.moves = new PoseMoves();
    this.landmarker = null;
    this.running = false;
    this.lastVideoTime = -1;
    this.status = "off";
  }

  async start() {
    if (this.running) return;
    this.status = "starting camera";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    this.video.srcObject = stream;
    await this.video.play();
    if (!this.landmarker) {
      this.status = "loading pose model";
      const { FilesetResolver, PoseLandmarker } = await import(`${VISION}/vision_bundle.mjs`);
      const fileset = await FilesetResolver.forVisionTasks(`${VISION}/wasm`);
      this.landmarker = await PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL, delegate: "GPU" },
        runningMode: "VIDEO",
        numPoses: 1,
      });
    }
    this.running = true;
    this.status = "stand in the middle, then press Calibrate";
    this.calibrate();
    requestAnimationFrame(() => this.loop());
  }

  stop() {
    this.running = false;
    this.status = "off";
    for (const track of this.video.srcObject?.getTracks() ?? []) track.stop();
    this.video.srcObject = null;
  }

  calibrate() {
    this.moves.startCalibration(performance.now() / 1000);
  }

  loop() {
    if (!this.running) return;
    const video = this.video;
    if (video.readyState >= 2 && video.currentTime !== this.lastVideoTime) {
      this.lastVideoTime = video.currentTime;
      const now = performance.now();
      const result = this.landmarker.detectForVideo(video, now);
      const landmarks = result.landmarks?.[0] ?? null;
      for (const move of this.moves.update(landmarks, now / 1000)) this.onMove(move);
      this.draw(landmarks);
      if (this.moves.calibrating) this.status = "calibrating: stand still in the middle";
      else if (!this.moves.ready) this.status = "can't see you: step back until hips and shoulders are in view";
      else this.status = `tracking: ${["left", "centre", "right"][this.moves.lane]}${this.moves.crouched ? ", squat" : ""}${this.moves.airborne ? ", jump" : ""}`;
    }
    requestAnimationFrame(() => this.loop());
  }

  // The preview is mirrored by CSS (so it behaves like a mirror); draw in image
  // coordinates and let the same CSS flip the overlay.
  draw(landmarks) {
    const c = this.overlay;
    c.width = this.video.videoWidth || 640;
    c.height = this.video.videoHeight || 480;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    const base = this.moves.base;
    if (base) {
      // The two lines the body crosses to change lane (mirrored back to image x).
      ctx.strokeStyle = "rgba(255,255,255,0.5)";
      ctx.setLineDash([8, 8]);
      for (const side of [-1, 1]) {
        const x = (1 - (base.bodyX + side * this.moves.opt.lane_enter * base.shoulderWidth)) * c.width;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, c.height);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    if (!landmarks) return;
    ctx.strokeStyle = "#8A7DFF";
    ctx.lineWidth = 4;
    for (const [a, b] of BONES) {
      const p = landmarks[a], q = landmarks[b];
      if (!p || !q) continue;
      ctx.beginPath();
      ctx.moveTo(p.x * c.width, p.y * c.height);
      ctx.lineTo(q.x * c.width, q.y * c.height);
      ctx.stroke();
    }
  }
}
