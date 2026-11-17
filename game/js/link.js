// The WebSocket between this page and surfer/server.py (on Pupper, or on the
// laptop with --sim).
//
//   page  -> robot   {"type": "control", "action": "start" | "stop" | "lie_down" | "stand_up", ...}
//                    {"type": "state", ...}      the game's state, a few times a second
//                    binary                      a JPEG of Pupper's view (stream source only)
//   robot -> page    {"type": "move", "move": "left", "duration": 0.7}   a move Pupper just started
//                    {"type": "ready", "ok": true, "message": "..."}     answer to "start"
//                    {"type": "status", ...}     what the robot is doing
//                    binary                      a JPEG of what Pupper sees, annotated
//
// If the page goes away, the robot notices the silence and stands still.

export class RobotLink {
  constructor(handlers) {
    this.handlers = handlers; // { onMove, onReady, onStatus, onDebugImage, onOpen, onClose }
    this.socket = null;
    this.connected = false;
    this.connect();
  }

  connect() {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${scheme}://${location.host}/ws`);
    socket.binaryType = "blob";
    this.socket = socket;
    socket.onopen = () => {
      this.connected = true;
      this.handlers.onOpen?.();
    };
    socket.onclose = () => {
      this.connected = false;
      this.handlers.onClose?.();
      setTimeout(() => this.connect(), 1500);
    };
    socket.onerror = () => socket.close();
    socket.onmessage = (event) => {
      if (typeof event.data !== "string") {
        this.handlers.onDebugImage?.(event.data);
        return;
      }
      const message = JSON.parse(event.data);
      if (message.type === "move") this.handlers.onMove?.(message);
      else if (message.type === "ready") this.handlers.onReady?.(message);
      else if (message.type === "status") this.handlers.onStatus?.(message);
    };
  }

  send(message) {
    if (this.connected) this.socket.send(JSON.stringify(message));
  }

  sendFrame(blob) {
    // Skip a frame rather than queue it: an old picture is worse than none.
    if (this.connected && this.socket.bufferedAmount < 200000) this.socket.send(blob);
  }
}
