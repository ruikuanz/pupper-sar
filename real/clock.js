// Slow motion for the game, without touching the game.
//
// real/play.py puts this script into its own two game windows before the page
// loads. It changes one thing: the clock the game reads. A game works out how
// far everything moved from "what time is it now?", so if its clock runs at
// 0.35 of real speed, the whole run plays at 0.35 of real speed: the same
// track, the same rules, the same pictures, only slower. Nothing of the game
// is read, copied or changed.
//
// Only the clock is scaled, not the browser's "draw the next picture" signal,
// so the game still draws about 60 pictures a second and slow motion is smooth.
// (Measured on 2026-11-15: with that signal scaled too, the game drew 18
// pictures a second at 0.3 speed.) Sound is not slowed: use --mute.
//
// It also tells play.py which arrow keys the player presses (window.gameKey),
// so a person's play can be recorded: see the Recorder in play.py.
(() => {
  if (window.__gameClock) return;
  if (window !== window.top) {
    // Inside the page's frames (the game is in one of them): the clock.
    const realNow = performance.now.bind(performance);
    let scale = __SPEED__;
    let baseReal = realNow();
    let baseGame = baseReal;
    const gameNow = () => baseGame + (realNow() - baseReal) * scale;
    const dateZero = Date.now() - baseReal;
    window.__gameClock = {
      // Change speed from now on (time already passed stays passed: the clock never jumps).
      set(speed) {
        const real = realNow();
        baseGame += (real - baseReal) * scale;
        baseReal = real;
        scale = speed;
      },
      get: () => scale,
    };
    performance.now = gameNow;
    Date.now = () => Math.round(dateZero + gameNow());
  } else {
    window.__gameClock = { set() {}, get: () => 1 };
  }
  // Every frame of the page: which arrow key went down.
  window.addEventListener("keydown", (event) => {
    if (!event.repeat && event.key.startsWith("Arrow") && window.gameKey) window.gameKey(event.key);
  }, true);
})();
