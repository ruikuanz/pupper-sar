# The real game: a person on the left, Pupper on the right

Lane Hound (`game/`) is a runner we wrote ourselves so that Pupper could read
it. This folder is the other way round: the game is **Subway Surfers itself**,
untouched, and everything else has to cope with it.

```
 ┌──────────────────────────┬──────────────────────────┐
 │ HUMAN                    │ PUPPER                   │
 │ the real game            │ the real game            │
 │ arrow keys or webcam     │ its picture -> Pupper    │
 │                          │ Pupper's moves -> keys   │
 ├──────────────────────────┴──────────────────────────┤
 │ start both runs · game speed · webcam · record my   │
 │ play · let Pupper play                              │
 └─────────────────────────────────────────────────────┘
```

Both games run in **slow motion** (0.35 of full speed unless you say
otherwise), so that the dog's body can keep up with the runner on screen.

Nothing of the game is copied or rebuilt in this repository: no art, no
levels, no code, not even a screenshot. `real/play.py` opens the game's own
web page, twice, and works the two windows the way two players would.

## Where it stands (read this first)

| Part | State |
| --- | --- |
| Two real games side by side, human left, Pupper right | works |
| Keys to each window at the same moment, whichever has the focus | works |
| "Start both runs" (title screen and score screen) | works |
| Human plays by keyboard | works (it is just the game) |
| Human plays by webcam (step, jump, squat) | wired up; the pose reader is Lane Hound's and is tested there; not yet tried in front of a camera on this game |
| Pupper's picture to the robot and its moves back as keys | works, about 29 pictures a second, tested with the pretend robot |
| Slow motion, the same in both windows, changed while playing | works |
| Recording a person's play, training a model from it, Pupper playing from the model | works end to end; how well a model plays is not measured yet (see "Teach Pupper") |
| **Pupper reading the real game well enough to survive** | **not with the colour rules: about 2 game seconds a run, slow motion or not. With a model: depends on your recording, not measured yet** |
| Any of it on the real Pupper | not run yet |

The last-but-one row is the honest headline. Slow motion gives the dog's body
the time it needs; it does not make the colour rules see any better. What can
is a model trained on your own play: the steps are under "Teach Pupper", and
the first step is yours.

## Run it

Once, on the laptop (it uses the Chrome you already have; nothing is downloaded):

```bash
pip install -r requirements.txt -r real/requirements.txt
```

Laptop only, with a pretend Pupper:

```bash
python3 surfer/server.py --sim --game real
```

```bash
python3 real/play.py
```

With the robot: on Pupper `surfer/start.sh --game real`, on the laptop the
usual tunnel (`ssh -N -L 8770:localhost:8770 pi@pupper<N>.local`) and then
`python3 real/play.py`. Just the two games and the webcam player, no Pupper:
`python3 real/play.py --no-pupper`.

`python3 real/play.py --speed 0.5` sets how fast runs play (1 is the game's own
speed, the default is 0.35); the slider in the strip changes it while playing.

Then, in the strip at the bottom:

1. **The first time only**: each window shows the game's tutorial. Click a
   window and play it through with the arrow keys (half a minute each). The
   windows have their own Chrome profile (`real/.chrome-profile`, not in git),
   so this is remembered, and nothing touches your everyday browser.
2. **Use webcam**, stand where the camera sees your hips and shoulders,
   **Calibrate**. Or skip it and use the arrow keys in the left window.
3. **Let Pupper play**: Pupper stands up and from then on plays whenever a run
   is on its screen. **Stop** stops it; so do the joystick, a fall, and closing
   the windows (the same safety as Lane Hound: `surfer/server.py`).
4. **Start both runs** starts the two games at the same moment. Press it again
   after a game over (it waits out the "Save me!" box, whose buttons it never
   clicks: one spends the game's keys and the other plays an advert).

## Slow motion

Pupper needs 0.8 s to shuffle sideways and over a second to jump. The game's
runner changes lane in about 0.3 s and its jump lasts 0.7 s. At the game's own
speed the dog can only trail behind the runner; at 0.35 of it, the runner's
moves take about as long as the dog's.

`real/clock.js` does it, and it does not touch the game. A game works out how
far everything has moved from "what time is it now?". `play.py` puts that
script into its own two windows, where it makes the clock the game reads run
slow. Same track, same rules, same pictures, only slower, and still about 56
pictures a second, so it is smooth. Both windows get the same speed, so the
race stays fair. The clock is only slow while a run is on screen: the menus
and the "Save me!" countdown pass at normal speed.

Everything on Pupper's side that is timed against the game (when to jump, how
long a move keeps the runner busy) counts **game seconds**: `play.py` tells
the server how fast the clock is running.

## Teach Pupper (record, train, play)

The colour rules in `real_perception.py` do not survive this game (numbers at
the end). The other way is the one a person detector takes: do not describe what
to look for, show examples. The examples are you, playing the left window.

1. **Record.** Start `python3 real/play.py` (no Pupper needed: add
   `--no-pupper`), press **Record my play**, and play runs in the left window,
   by arrow keys or by webcam. Slow motion helps here too: you crash less, and
   your key presses land close to the moment that called for them. Every
   picture and every key is saved under `real/recordings/` (not in git: they
   are pictures of the game). The strip shows how much is saved. Aim for an
   hour of play; ten minutes is enough to see whether it is learning.
2. **Train**, on the laptop:

   ```bash
   pip install -r real/requirements-train.txt
   ```

   ```bash
   python3 real/train.py
   ```

   It writes `real/model.npz` and prints, for runs it kept aside, how many of
   your moves the model also made and how many extra ones it made.
3. **Play.** Give the server the model:

   ```bash
   python3 surfer/server.py --sim --game real --model real/model.npz
   ```

   On Pupper: copy the file over (`scp real/model.npz pi@pupper<N>.local:pupper-sar/real/`)
   and `surfer/start.sh --game real --model real/model.npz`. Pupper needs only
   numpy and OpenCV for it. The picture in the strip then shows a bar for each
   move: how sure the model is that you would press it now.

What has been tried (2026-11-15), and what has not. With no person's
recording to hand, the colour-rules player stood in for the person in the left
window: 51 runs, 3.7 game minutes, 119 key presses, recorded through the same
recorder. `train.py` made a model from them in 12 s. On the runs kept aside it
made 7 of the stand-in's 16 moves and 17 extra ones, and through
`server.py --model` it played the right window the way its teacher does: runs
of 2 to 4 game seconds, the simulated dog following every move. That shows the
chain works on the real game's pictures. It says nothing about how well Pupper
will play after learning from a good player with twenty times the material:
that is the measurement still to make, and it needs your recording.

What it learns is "press what this person pressed when the picture looked like
this". So play the way you want Pupper to play: dodge early, do not chase
coins into trains. The last second before every crash is left out of the
lessons. When the game changes city, record a little more and train again.

## How it works

| File | What it does |
| --- | --- |
| `real/play.py` | Laptop. Opens the three windows, sends keys to the two games, sends Pupper's picture to the server, shows the strip, slows the games, records the person. |
| `real/clock.js` | The slow-motion clock (and the report of which arrow keys were pressed, for the recorder). |
| `real/train.py` | Laptop. Recordings -> `real/model.npz`. Needs PyTorch. |
| `game/real.html`, `game/js/real_panel.js` | The strip. The webcam reader is `game/js/pose_moves.js`, unchanged. |
| `surfer/server.py --game real` | Pupper (or `--sim`). Same server, same safety; the eyes and the decisions are swapped. |
| `surfer/real_perception.py` | One picture -> the striped boards ahead, and how far each lane is open. |
| `surfer/real_policy.py` | That -> jump, squat, left, right, or keep running. |
| `surfer/real_learned.py` | The other eyes (`--model`): one picture pair -> how likely each move is, from the trained model. numpy only. |
| `tests/test_real.py` | 82 desk checks on pictures the test draws itself, including a recording, a training run and playing from the result. |

Three things are different from Lane Hound, all forced by the game:

- **The key goes first, the dog follows.** The game's runner changes lane in a
  tenth of a second; the dog needs 0.8 s to shuffle and over a second to jump.
  So when Pupper decides, the key goes to the game at once and the same move
  is put on a list for the body (`body_queue`). The dog makes every move, a
  little behind; left-then-right cancels out, and if it falls more than three
  moves behind it skips tricks but never a lane change, so it ends up in the
  lane the runner is in.
- **Stream only.** Pupper sees the game as pictures sent over the network.
  Its fisheye camera cannot read this game off a screen: a board that is half
  a second away is 30 pixels wide in the game's own picture.
- **Two windows, not two frames in one page.** The game's page does not allow
  being put inside another page, and that is its right.

### What Pupper looks for

- **Striped boards**, by their red: a low wide board with a chevron (jump), a
  tall one (roll: the dog squats), a low one with upright stripes (the end of a
  track: go around). How far: by how wide the board looks, the way a person
  follower judges distance by the height of a box.
- **Open track**: the colour of the track beside the runner's legs is "open
  track" for this picture. Each lane is followed up the picture until it stops
  looking like that. Whatever stops it is a train.

## What is not good enough yet

Measured on 2026-11-15: 10 runs of the real game, played by
`real_perception.py` and `real_policy.py` with no dog in the loop (keys only,
so this is the reader on its own, at its best). Seconds each run lasted:

    4.4  1.4  2.0  6.6  2.2  2.2  1.5  1.2  4.1  1.4

Median 2.1 s, best 6.6 s.

The same reader with the game at half speed (6 runs, in game seconds):

    1.3  1.3  4.3  2.4  1.2  1.3

No better. Slow motion gives twice the time to act on what is seen; it does
not change what is seen, and what the colour rules see is wrong. One of these
runs, looked at picture by picture: the run starts beside a parked train whose
front is painted in red and yellow panels, the reader takes a panel for a
striped board in the next lane and the pink-lit track for a wall, and its very
first move steers the runner in front of the train. Leaving the keys alone
would have lasted longer.

The reader was written against one city of the game (it changes every few
weeks) in an afternoon, and that city is hard on it: the trains are the same
maroon as the track, some are painted with red and yellow panels that pass
for boards, and the tunnels are lit pink. Each of those ended runs. The
numbers in `real_perception.py` that carry the timing (`WIDTH_AT_RUNNER`,
`CLOSING_RATE`) come from a handful of recordings.

A hand-tuned colour reader is the wrong tool for this game. The right one is
built and waiting for its lessons: see "Teach Pupper" above. What is known
about it so far is at the end of that section's table row and in
`tests/test_real.py`; what is not known yet is how long Pupper survives with a
model trained on a person's play.

## Things to know

- **Which game.** This is the current browser release, on the publisher's own
  portal (`GAME_URL` in `play.py`). The 2012 build is not offered anywhere
  official, and this repository will not fetch a copy from elsewhere. If you
  have a legitimate one that runs in a browser, point `GAME_URL` and
  `GAME_ELEMENT` at it; the layout numbers in `real_perception.py` will need
  measuring again.
- **The page's terms.** A program is pressing keys on a web game here. That
  is a class demo on your own machine, but the site's terms of use are yours
  to read and keep; the page is left as it is, adverts included.
- **Sound** comes from both games, and slow motion does not slow it: the music
  plays at its own speed over a slow run. `--mute` silences both.
- **Slow motion and the game's own records.** A run played at 0.35 speed is
  not the game as its makers time it. Keep slowed scores to yourself: do not
  put them forward anywhere scores are compared.
- **Not tested on screen.** Everything above was tested with Chrome running
  without windows (`--headless`). The first run with windows is the first time
  the layout (two halves and a strip) is seen: `--screen W H` sets the size by
  hand if it guesses wrong.
