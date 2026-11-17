# Pupper SAR

A small quadruped robot, 
[Pupper v3](https://github.com/Nate711/pupperv3-monorepo), that understands
 natural language, can play Subway Surfers, and can echolocate.

1. **Natural Language and Interaction** A vision language model
   hears you, sees through Pupper's camera, talks back, and moves the
   robot through a set of checked tools: walk, turn, arc, follow a
   person, and fourteen tricks played by trained policies.
2. **Subway Surfers** It plays the video game by watching a
   screen, deciding, and running in place, sidestepping, jumping and squatting to steer the
   runner, racing a person who plays the same course next to it by moving in
   front of a webcam.
3. **Echolocation** It chirps from its own speaker, listens
   with wireless microphones clipped to its body, and from the echoes alone
   draws a map of the room and walks around in it without using its camera.

All three use one robot stack in this repository. Nothing else needs to
be cloned: the laptop side needs Python (and Node for the assistant's web app and some tests), and the
robot side needs only the ROS 2 software a Pupper v3 ships with.

---

## What's inside

| Area | Where | What it does |
| --- | --- | --- |
| **Embodied vision-language agent** | `gemini/` | Gemini Live hears, sees through the robot's camera and talks back. The tool layer is its only way to move the robot and its only source of truth about what happened |
| **Tool design for a model** | `gemini/tools/`, `gemini/prompt.md` | Each tool specifies the robot's physical limits; every call is checked against its declaration, and refusals name the limit so the model can retry |
| **Concurrency and robot state** | `gemini/robot_api.py` | A job queue (a human and thus the model can ask for several things in one breath), a state machine of the hardware controller, and comprehensive safeguards |
| **Real-time computer vision** | `surfer/perception.py`, `screen_finder.py` | Reads a game from a screen: color segmentation, distance from where an object meets the floor, tracking rows between frames, measured track speed |
| **Camera geometry** | `surfer/screen_finder.py` | Undoes the fisheye (double-sphere model), finds the on-screen frame, recovers a homography, and works out where the robot stands from the frame's four corners |
| **Legged-robot control** | `surfer/moves.py`, `pupper/` | Five body moves timed for the game, executing via a live learned walking policy, so there is no controller switch between moves |
| **Kinematics** | `surfer/squat_reference.py` | Forward and inverse kinematics of all four legs, used to build a squat the walking policy plays the way it plays its trained jump |
| **Learning from demonstration** | `surfer/real_learned.py`, `real/train.py` | A CNN trained on a human's own play (recorded pictures and key presses) that runs on the robot with numpy alone |
| **Human pose estimation** | `game/js/pose_camera.js`, `pose_moves.js` | MediaPipe Pose in the browser, turned into game moves, measured against each player's own calibrated body |
| **Acoustic signal processing** | `echo/chirp.py` | Linear-FM chirps, matched filtering (pulse compression) to about 7 cm, timing that cancels unknown audio and radio delays |
| **Geometric inference** | `echo/ears.py` | Recovers walls from echo times with the image-source model, for two microphones (moving) or four (one ping) |
| **Mapping and localization** | `echo/room_map.py` | Builds a wall-and-corner map from echoes and uses it to correct the robot's own position after every step |
| **Safe autonomy** | `echo/navigator.py`, `pupper/pupper_ros.py` | Walks only where echoes say is free, fall detection, e-stop handling |
| **Systems** | `surfer/server.py`, `echo/server.py`, `real/play.py` | Async servers on the robot, WebSocket streaming, browser automation of a commercial game, a game clock slowed without touching the game |
| **Game design for a robot** | `game/` | An original three-lane runner drawn so a camera can read it, with seeded courses that are fair to both players |

---

## Part 1: Natural Language and Interaction

Talk to Pupper, and it listens, looks and does what you ask. Gemini Live is the
brain: a foundation model that hears you, sees through the camera and talks
back, in real time. Trained policies are the physical controllers. In between are the
**tools**, the only way Gemini can move the robot and the only way it learns
what really happened.

```
 browser (laptop, or Pupper's own screen)                 Pupper
┌───────────────────────────────────────┐   GET /tools   ┌──────────────────────────────────────┐
│ gemini/web/  React app                │ ◀───────────── │ gemini/server.py   port 8000          │
│  mic + camera ──▶ Gemini Live (cloud) │  POST /tools/x │   tool_registry.py  check every call  │
│  Gemini calls a tool ────────────────────────────────▶│   tools/            the tools         │
│  ◀── {"success", "message"} ──────────────────────────│   robot_api.py      queue, legs, falls│
│  speaker ◀── Gemini speaks            │                └──────────────┬───────────────────────┘
│  eyes follow people (detections)      │                               │ /llm_cmd_vel, controller switches
└───────────────────────────────────────┘                               ▼ pupper/  (walking + trick policies)
```

| Tool | What it does |
| --- | --- |
| `stand_up`, `lie_down`, `stop`, `robot_status` | the basics; `stop` cancels everything at once, even requests still being handled |
| `move`, `turn`, `arc` | walk or run, turn in place, walk along part of a circle |
| `follow` | walk after the nearest person (the platform's Hailo person detector); any motion ends it |
| `play_trick` | one of 14 tricks, each a trained tracking policy (sit, push-ups, downward dog, swim, sneeze, ...) |
| `interactive_action` | actions inside an interactive trick (e.g. shake a paw while sitting), once one is installed |

**What the model reads is what the robot enforces.** Every speed, distance and
radius limit in a tool's description is the same number the code checks
(`MoveLimits`), so Gemini can plan moves the robot accepts. A refusal names the
limit ("too tight: at 0.4 m/s the radius must be at least 0.21 m"), so it can
retry. Rules that matter are enforced in code as well as in the prompt: any
motion ends follow mode (and the reply says so, so Gemini can tell the person
first), a fallen robot refuses to move, and a stop clears requests that are
still being handled.

**A queue, because people ask for several things at once.** "Sit, then shake"
is two tool calls in one breath. Each tool replies as soon as its job is
*accepted*, and the jobs run one at a time. Each request is checked twice: once
against the state Pupper *will* be in when the queue reaches it, and again
when the job runs, in case a fall or a stop came in between.

**Tricks are policies, not animations.** Every trick is a tracking policy
trained in simulation to follow a reference motion while keeping its balance,
loaded as its own `ros2_control` controller. So the robot is always in exactly
one simple state: walking, one trick, motors idle, or fallen. Interactive
tricks are trained on a clip library and take requests while Pupper stays in
them. A trick that walks in its own stance hands the legs to its own walking
policy, with its own speed limits. New tricks are fetched from training runs
with `gemini/get_trick.py`.

**Seeing.** The fisheye frame is undistorted to an equirectangular view before
Gemini sees it, so people look like people instead of bending around the edge
of the lens. Pupper's on-screen eyes follow people from the person detector.

Credit: Gemini Pupper was created by [Peng Xu](https://scholar.google.com/citations?user=460NWeQAAAAJ&hl=en)
(Google DeepMind), original repo [sippeyxp/gemini-pupper](https://github.com/sippeyxp/gemini-pupper).

---

## Part 2: Subway Surfers

Two views of the same course are on a laptop screen. A person plays one by
moving in front of the webcam. Pupper plays the other with its body.

| Game move | Pupper | Person |
| --- | --- | --- |
| nothing | runs in place | jogs on the spot |
| left / right | sidesteps one lane (three marks on the floor) | steps left / right |
| down | squats | squats |
| up | jumps | jumps |

There are two games to play:

- **Lane Hound** (`game/`), an original runner written for this project. Owning
  the game makes the rest possible: both views are the same seeded course, the
  runner on screen moves at the speed Pupper's body really can, and every
  obstacle kind is drawn in one saturated colour, so a hue test is the
  detector.
- **Subway Surfers**, the real browser release, untouched (`real/`).
  `real/play.py` opens the publisher's own page twice and works the two windows
  the way two players would. Nothing of the game is copied into this
  repository. See [`real/README.md`](real/README.md).

### How it works

```
 laptop screen                                Pupper (or --sim)
┌──────────────────────────────┐   picture   ┌────────────────────────────────────┐
│ two views, one course        │ ──────────▶ │ see    screen_finder.py  (camera)  │
│  ┌────────┐  ┌────────┐      │  camera or  │        perception.py   colour+floor │
│  │ HUMAN  │  │ PUPPER │      │  WebSocket  │ decide policy.py       which move   │
│  │        │  │ (green │      │             │ act    moves.py        the 5 moves  │
│  │        │  │ frame) │      │ ◀────────── │ report server.py                    │
│  └────────┘  └────────┘      │    moves    └────────────────────────────────────┘
│   ▲ webcam + MediaPipe       │                 │ /llm_cmd_vel, /sar_walk/reference_trick
└───┼──────────────────────────┘                 ▼  pupper/  (walking policy, safety)
  person
```

**Seeing.** Everything in the game stands on the same floor, and the floor
recedes upward in the picture. So the bottom edge of an obstacle tells how far
away it is, and distance over track speed is time to impact. `perception.py`
walks up each lane's centre line and reads off runs of colour. It knows the
one object whose bottom edge is *not* on the floor (an overhead bar) and
corrects for it. `RowTracker` follows rows from frame to frame and measures
the track's speed, so timing stays right as the game speeds up.

**Deciding.** `policy.py`: change lane as early as possible (it is slow), and
jump or squat at exactly the right moment (the body has to be in the air when
the hurdle passes). The timing comes from the moves Pupper can really make.

**Moving.** The fast part is that all five moves go through one running
controller, a learned walking policy, so nothing has to stand up between
moves (that takes 4 seconds):

- *Run in place*: with no forward speed and a small command, the policy steps
  on the spot. A small sideways speed that flips sign twice a second keeps the
  feet stepping and the body still.
- *Sidestep*: the same gait with a real sideways speed, for a fixed time.
- *Jump*: a table of joint angles the policy was trained to follow. It plays
  the table once and keeps its own balance while doing it.
- *Squat*: a table of the same kind that we build ourselves with forward and
  inverse kinematics. Every foot moves 4.5 cm straight up toward the body and
  back, staying under the same spot, so the body stays level and in its lane.

**Two ways for Pupper to see** (`--source`):

- `stream`: the page sends Pupper's view over a WebSocket. Nothing to line up.
- `camera`: Pupper looks at the screen with its own fisheye camera.
  `screen_finder.py` remaps the fisheye to a pinhole view (straight lines must
  be straight before a rectangle can be found), finds the green frame around
  Pupper's view, warps the inside back to the game's own picture, and from the
  frame's four corners works out where Pupper stands and which way it faces.
  With `--hold` it uses that to steer back to its lane instead of trusting
  dead reckoning.

**The person.** MediaPipe Pose runs in the browser. `pose_moves.js` turns
shoulders and hips into moves, measured in units of the person's own shoulder
width and torso length from a calibration pose, so it works for anyone at any
distance.

**The real game.** Subway Surfers defeats hand-tuned colour rules within
seconds (`real/README.md` has the measured numbers and why). So there is a
second approach: record a person's play, the pictures and the keys, and train
a small CNN (two stacked frames 0.1 s apart → none / left / right / jump /
squat) on it with PyTorch on the laptop. The result is a file of numbers that
Pupper runs with numpy alone. The training labels around each key press
ignore a short window on either side, so the network is not punished for
being a moment early or late. Every frame is also used mirrored, with left
and right swapped. And because a dog is slower than a cartoon runner,
`real/clock.js` slows the game by scaling the clock the page reads. The game
itself is not changed, it still draws at 60 frames a second, and both windows
get the same speed so the race stays fair.

---

## Part 3: Echolocation

| Mode | Ears | One look is | Run with |
| --- | --- | --- | --- |
| 1 | two DJI Mic 3 transmitters + the compact USB-C receiver | three pings, turning 60° between them | `--ears two` |
| 2 | the main receiver + up to four transmitters | one ping, no turning | `--ears four` |

```
 speaker ── chirp ──▶ air, walls ──▶ ears (DJI mics on the body)
                                        │ one recording per ear
   chirp.py     when did the chirp arrive?  the direct sound, then each echo
   ears.py      which wall would give these arrival times in every ear?
   room_map.py  walls and corners in the room; where is Pupper among them?
   navigator.py listen ─▶ decide ─▶ turn ─▶ walk a step ─▶ listen again
        │ /llm_cmd_vel                       ▲ heading from the IMU
        ▼  pupper/  (walking policy, safety)
```

**Time is distance (`chirp.py`).** Sound travels at 343 m/s. The chirp sweeps
3 to 13 kHz in 8 ms. Sliding a copy of it along the recording (a matched
filter) squeezes it into a spike about 7 cm of travel wide: pulse compression.
Each echo is timed from the *direct* sound in the same microphone, so the
unknown delays of the sound card and the radio link cancel out, and the
microphones need not be in step with each other. That is why they ride on the
dog and not around the room.

**Triangulation (`ears.py`).** A flat wall is a mirror: its echo arrives as if
from a second speaker behind the wall (the image source). Each ear's arrival
time puts that image on a circle around the ear. The circles cross at the
image, and the wall is the perpendicular bisector between the speaker and its
image. Two ears' circles cross twice (the wall and a mirror twin), so mode 1
turns and listens again: the wall stays put in the room and the twin moves.
Three or more ears not in a line cross once, so mode 2 needs one ping. In code
this is one score for every candidate wall: did every ear, on every ping, hear
something at the moment this wall would give? (The idea is from Dokmanić et
al., *Acoustic echoes reveal room shape*, PNAS 2013, shrunk onto a dog.)

**The map (`room_map.py`).** Distances from echoes are good to a couple of
centimetres. Bearings from ears a hand apart are only good to about ten
degrees. So the map leans on distances: a wall's distance changes by exactly
how far Pupper moved toward it, while a corner's is the distance to one point.
Fitting both tells which one it is and where. The same echoes correct Pupper's
position after every step (it walks short; the walls do not move): a small
SLAM loop built on range data alone.

**Walking blind (`navigator.py`).** A state machine on a 20 Hz clock that never
blocks. Every echo is treated as a face-on wall that runs on forever, so a step
is never longer than the echoes say is free, and the body keeps 0.3 m from
everything it hears.

**Simulation (`sim_room.py`, `dog.py`).** The test room has delays, noise,
ringing, a coloured speaker response, and floor and ceiling bounces. The
simulated dog walks 15 % short, its gyro drifts, and its microphones sit up to
a centimetre from where `ears.yaml` says. In a 4 × 3 m room, after eight
looks, both modes draw the corners to within 20 cm, know their own position to
within 15 cm, and never come closer than 0.25 m to a wall.

---

## The robot layer (`pupper/`)

All three parts run on one launch file, `pupper/launch/pupper.launch.py`, and
drive one walking controller. The game and echolocation talk to the robot
through `pupper/pupper_ros.py`. The assistant uses its own, fuller
`gemini/robot_api.py` (queue, tricks, follow mode) on the same stack. Only one
of the three servers runs at a time, and each start script refuses while another
is running.

- **One controller holds the legs.** Standing up is a single strict
  `ros2_control` switch: our walking controller (`sar_walk`) on and every other
  leg controller off, in one request, so two policies never command the motors
  at once.
- **Our own walking controller.** `walk_controller.py` copies the robot's
  installed walking-policy settings at launch time (gains, policy file, tilt
  limit) and turns on reference-trick playback. The same code runs on whatever
  policy the robot has.
- **Falls.** A controller that tilts past its limit latches an e-stop that no
  topic reports. We catch its log line on `/rosout`, stop and refuse to move.
  Standing up again first checks the IMU: level, and still for a full second,
  so it won't start a policy while Pupper is on its side or in someone's hands.
- **E-stop.** The joystick's e-stop relaxes the platform's controllers. We relax
  ours too.

- **Assistant mode.** `stack.sh start --assistant` also loads every trick
  controller at launch and makes sure the platform's person follower runs.

---

## Try it on a laptop (no robot)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**The game.**

```bash
.venv/bin/python surfer/server.py --sim
```

Open http://localhost:8770 and press **Start**. The right view is played by
the simulated Pupper: the page streams that view to the server, the server's
vision reads it and decides, and moves come back with the timing of the real
dog's. The left view is yours: arrow keys, or **Use webcam** (stand so your
hips and shoulders are in view, press **Calibrate**, then step, jump and
squat).

**Echolocation.**

```bash
.venv/bin/python echo/server.py --sim --ears two
```

Open http://localhost:8771, press **Stand up**, then **Explore**. The dark
outline is the map Pupper builds from echoes. The faint dashed lines are the
real (simulated) room and dog, which Pupper is never told. Click the map to
send it somewhere. Other rooms: `--room arena`, `slanted`, `post`.

**The assistant.**

```bash
cd gemini/web && npm ci && npm run build && cd ../..
```

```bash
.venv/bin/python gemini/server.py --sim
```

Open http://localhost:8000, open **Settings**, paste a Gemini API key (from
[Google AI Studio](https://aistudio.google.com/apikey); it stays in your
browser), and press **WAKE PUPPER**. The robot is pretend: nothing moves, and
the debug view shows every tool call and its result. Or talk in text, with
every tool call printed:

```bash
GEMINI_API_KEY=... .venv/bin/python gemini/chat.py
```

**The real game** (needs Chrome): see [`real/README.md`](real/README.md).

```bash
.venv/bin/pip install -r real/requirements.txt
```

```bash
.venv/bin/python real/play.py
```

## Tests

```bash
.venv/bin/python tests/test.py        # vision, policy, moves, squat table, simulated fisheye camera (61 checks)
.venv/bin/python tests/test_echo.py   # chirp, triangulation, map, navigator (106 checks, ~10 s)
.venv/bin/python tests/test_real.py   # the real game's reader, learned player, recorder, clock (82 checks)
.venv/bin/python tests/test_pupper.py # the robot layer against a pretend ROS: switches, falls, e-stop (67 checks)
.venv/bin/python tests/test_assistant.py  # every tool, the queue, follow mode, tricks, stop, the prompt (165 checks)
.venv/bin/python tests/replay.py      # whole games through the vision, every frame checked
```

```bash
cd tests && npm install && cd ..
```

```bash
node tests/game_test.mjs              # game rules, course fairness, body-pose controls (22 checks)
```

```bash
node tests/play_headless.mjs --seconds 120 --seed 5   # real engine + renderer vs. the simulated robot (start surfer/server.py --sim first)
```

`play_headless.mjs` is the end-to-end check: it fails if Pupper crashes once,
or never uses one of the moves.

## On the robot

```bash
# on the Pupper
git clone https://github.com/ruikuanz/pupper-sar.git ~/pupper-sar && cd ~/pupper-sar
gemini/setup.sh                     # once: builds the app, asks for your Gemini key
gemini/start.sh --kiosk             # the assistant, on Pupper's own screen, mic and speaker
surfer/start.sh                     # or the game; or: surfer/start.sh --source camera
echo/start.sh                       # or echolocation (sudo apt install libportaudio2; pip install sounddevice)
```

```bash
# on the laptop
ssh -N -L 8000:localhost:8000 pi@pupper<N>.local    # 8770 for the game, 8771 for echolocation
```

Then open http://localhost:8000 (8770, 8771). It has to be `localhost`: browsers only give
the webcam to secure pages. `gemini/stop.sh`, `surfer/stop.sh` and `echo/stop.sh`
lie Pupper down and stop everything. Logs: `/tmp/pupper-gemini/`,
`/tmp/lane-hound/` and `/tmp/pupper-echo/`. If the
robot's ROS workspace is not at `~/pupperv3-monorepo/ros2_ws`, set
`PUPPER_ROS_WS`.

**Safety.** The assistant moves only through its checked tools, and "stop"
stops it. The game and echolocation move Pupper only between Start and Stop (or
during a Look, Explore or Go-to). It stops by itself when the page closes or goes quiet, when
it falls, and when the joystick's e-stop is pressed. The joystick outranks it.
Jumps need clear space. Keep a hand on the e-stop for anything new, on the
floor, never on a table.

## Layout

```
gemini/               the assistant: talk to Pupper
  server.py           the robot server: the tools, the camera, the web app (--sim: a pretend robot)
  prompt.md           who Pupper is and how it behaves (Gemini's system prompt)
  tools/              the tools Gemini calls: basics, walking, following, trick_tools, interactive
  tool_registry.py    turns @tool functions into what Gemini reads; checks every call
  robot_api.py        the queue, the controller switch, follow mode, tricks, falls, the e-stop
  tricks.py           reads tricks/<name>/trick.yaml       tricks/  the 14 trick policies
  web/                the browser app (React, Gemini Live): voice, camera, eyes, debug view
  chat.py             talk to Gemini in text, on the pretend robot or the real one
  kiosk.py            runs the app on Pupper's own screen    get_trick.py  fetch a newly trained trick
  setup.sh, start.sh, stop.sh
pupper/               the robot layer, shared by all three parts
  pupper_ros.py       the one file that talks to ROS: legs, falls, e-stop, camera, heading
  walk_controller.py  our instance of the walking policy and the trick controllers, and how they load
  launch/             the robot stack (pupper.launch.py) and its parameters
  stack.sh            start / stop the robot stack       ros_env.sh  the ROS environment
  references/         jump.json: the jump the walking policy was trained on
  camera_params.yaml  the fisheye model and a fixed white balance
game/                 Lane Hound, the game page (runs in any browser)
  config.json         shared by the page and the robot: projection, colours, timing
  js/engine.js        the rules: course, players, collisions (no DOM)
  js/render.js        one view; the projection perception.py inverts
  js/pose_*.js        webcam + MediaPipe -> moves     js/link.js  WebSocket to the robot
  real.html           the control strip for the real game
surfer/               the player: runs on Pupper, or on a laptop with --sim
  server.py           see / decide / act / report, and serves the game
  perception.py       picture -> rows of obstacles     policy.py   rows -> a move
  moves.py            the five moves; every tuning constant for the body
  screen_finder.py    camera picture -> game view + where Pupper stands
  squat_reference.py  FK + IK -> references/squat.json
  real_*.py           the real game: colour reader, its policy, the learned player
  robot_io.py         the real robot (through pupper/) and the simulated one
real/                 the real game on the laptop: play.py, clock.js, train.py
echo/                 echolocation
  chirp.py  ears.py  room_map.py  navigator.py      the chain, from sound to steps
  sound.py  dog.py  sim_room.py                     real and simulated sound card, robot, room
  server.py, page/    act / report, and the map page
  mic_check.py        run this first with real microphones
  ears.yaml           where the speaker and mics are on your Pupper (measure them)
tests/
```
