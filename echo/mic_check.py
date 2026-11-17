"""Check the ears before trusting them: the first thing to run with real microphones.

No robot code, no ROS: it only plays pings and listens. It runs on the Pupper
and on a laptop (plug the DJI receiver into the laptop and use its speakers:
the fastest way to hear your first echo is at a desk, a book for a wall).

    python3 echo/mic_check.py --list              # the sound devices, and how many channels each records
    python3 echo/mic_check.py --tap               # which channel is which mic: tap each mic in turn
    python3 echo/mic_check.py --ears two          # ping, and say what each ear heard and where the walls are
    python3 echo/mic_check.py --ears four --pings 8
    python3 echo/mic_check.py --ears two --wall 1.0   # calibrate: Pupper square-on to a wall 1.0 m ahead
    python3 echo/mic_check.py --sim --ears four   # the same report from the pretend room (no hardware)

Options for all: --mic NAME --speaker NAME (a device number or part of a name).

What to look for in the report
------------------------------
    loudness     the direct sound should be well above the noise (40 dB or
                 more) and nowhere near clipping. Set the speaker volume and
                 the receiver's gain so it is.
    steadiness   every chirp of a ping equally loud. If not: auto gain or
                 noise cancelling is still on (sound.py lists the settings).
    in step      the time between two ears hearing the direct sound, ping
                 after ping. We do not need the ears to be in step (chirp.py
                 says why), so this is only shown; a spread of milliseconds
                 just tells you the radio link buffers each mic separately.
    echoes       the same walls at the same distances on every ping, to a
                 centimetre or two. Move Pupper 10 cm toward a wall: that
                 echo should come 10 cm closer.

--wall: the one calibration
---------------------------
ears.yaml has where the speaker and mics are by ruler. Sound does not go
through the body, it goes around it, so the direct sound's path to each ear is
a little longer than the ruler says, and every wall would read a little far.
Put Pupper in the open, standing, nose square-on to a big flat wall, the
middle of its body exactly D metres from it (tape measure), and run --wall D.
It finds that wall's echo in each ear and writes the direct paths that make it
read D into echo/calibration.json, which ears.py uses from then on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import chirp
import ears as ears_module
from sim_room import SimRoom


def decibels(ratio: float) -> float:
    return 20.0 * float(np.log10(max(ratio, 1e-9)))


def tap_test(sound) -> None:
    """Record while the person taps each mic, and say which channel heard each tap loudest."""
    import sounddevice as sd

    seconds = 8
    print(f"Tap each mic once, a second apart, in the order you want to know them. Recording {seconds} s ...")
    heard = sd.rec(int(seconds * chirp.SAMPLE_RATE), samplerate=chirp.SAMPLE_RATE, channels=sound.ears,
                   device=sound.mic, dtype="float32")
    sd.wait()
    block = chirp.SAMPLE_RATE // 20
    blocks = np.abs(heard[: len(heard) // block * block]).reshape(-1, block, sound.ears).max(axis=1)
    loud = blocks.max(axis=1)
    taps = []
    for i in np.argsort(-loud):
        if loud[i] > 8 * np.median(loud) and all(abs(i - t) > 6 for t in taps):
            taps.append(int(i))
    if not taps:
        return print("No taps heard. Is the right device chosen (--mic), and the receiver not muted?")
    for n, i in enumerate(sorted(taps[:sound.ears * 2]), 1):
        channel = int(np.argmax(blocks[i]))
        print(f"  tap {n} at {i * block / chirp.SAMPLE_RATE:4.1f} s: channel {channel + 1} "
              f"(levels: {'  '.join(f'{v:.2f}' for v in blocks[i])})")
    print("The order of `mics` in ears.yaml must be channel 1, channel 2, ...")


def report(geometry, sound, pings: int, wall: float | None, save: str | None) -> bool:
    """Ping a few times and print what the ears made of it. Returns True if nothing looked wrong."""
    ears = ears_module.Ears(geometry)
    names = geometry.names
    looks, hearings, problems = [], [], []
    for n in range(pings):
        recording = sound.ping()
        if save:
            np.save(save if pings == 1 else f"{save}.{n}", recording)
        hearing = ears.listen(recording)
        hearings.append(hearing)
        looks.append(ears.locate([(0.0, hearing)]))
        problems += hearing.problems
        time.sleep(0.2)

    print(f"\n{geometry.mode} ears, {pings} pings on: {sound.name}")
    print(f"  {'ear':<12} {'direct sound':>13} {'above noise':>12} {'steadiness':>11}")
    for k, name in enumerate(names):
        level = np.mean([h.profiles[k].direct_level for h in hearings])
        noise = np.mean([h.profiles[k].noise for h in hearings])
        steady = min(h.profiles[k].steadiness for h in hearings)
        clipped = any(h.profiles[k].clipped for h in hearings)
        print(f"  {name:<12} {decibels(level):>9.0f} dBFS {decibels(1 / max(noise, 1e-9)):>9.0f} dB {steady:>10.0%}"
              + ("   CLIPPING" if clipped else ""))
    print("\n  in step? (when ear k heard the direct sound, after the first ear did)")
    for k in range(1, len(names)):
        gaps = np.array([h.profiles[k].direct_at_s - h.profiles[0].direct_at_s for h in hearings]) * 1000
        by_ruler = (geometry.direct_m[k] - geometry.direct_m[0]) / chirp.SPEED_OF_SOUND * 1000
        print(f"  {names[k]:<12} {gaps.mean():+7.2f} ms, spread {gaps.std():.2f} ms   (by ruler alone: {by_ruler:+.2f} ms)")

    two_ears = geometry.needs_turning
    print("\n  echoes (distance from the middle of the body; directions relative to the nose, + is left)")
    if two_ears:
        print("  Two ears, one heading: every wall also shows up mirrored in the line through the ears.")
    seen: list[list] = []   # the same echo across pings: [[echo, ...], ...]
    for look in looks:
        for echo in look.echoes:
            same = next((group for group in seen if abs(group[0].distance_m - echo.distance_m) < 0.08
                         and abs((group[0].bearing_deg - echo.bearing_deg + 180) % 360 - 180) < 25), None)
            (same if same is not None else seen.append([]) or seen[-1]).append(echo)
    steady_echoes = sorted((g for g in seen if len(g) >= max(1, pings // 2)), key=lambda g: g[0].distance_m)
    for group in steady_echoes:
        d = np.array([e.distance_m for e in group])
        print(f"  {d.mean():5.2f} m  (+-{d.std() * 100:.1f} cm)  at {np.mean([e.bearing_deg for e in group]):+5.0f} deg   "
              f"strength {np.mean([e.strength for e in group]):.2f}   in {len(group)} of {pings} pings")
    if not steady_echoes:
        print("  none. In a big open space that is right; near a wall it is not: check the settings in sound.py.")

    for line in sorted(set(problems)):
        print(f"\n  ! {line}")

    if wall is not None:
        calibrate(geometry, ears, hearings, wall)
    return not problems


def calibrate(geometry, ears, hearings, wall: float) -> None:
    """--wall D: set each ear's direct path so a wall D m dead ahead reads D."""
    # The echo of a wall `wall` m ahead of the body centre comes from the speaker's mirror image in it.
    image = np.array([2 * wall - geometry.speaker[0], geometry.speaker[1], geometry.speaker[2]])
    expected = np.linalg.norm(geometry.mics - image, axis=1)
    new = []
    print(f"\n  calibrating against a wall {wall:.2f} m dead ahead")
    for k, name in enumerate(geometry.names):
        found = []
        for hearing in hearings:
            path, value = hearing.path_m[k], hearing.reflectivity[k]
            near = np.abs(path - expected[k]) < 0.20
            if near.any() and value[near].max() > ears_module.MIN_REFLECTIVITY:
                found.append(path[near][np.argmax(value[near])])
        if len(found) < max(1, len(hearings) // 2):
            return print(f"  {name}: no echo near {expected[k]:.2f} m of path. Is the wall big, flat, square-on and "
                         f"{wall:.2f} m from the middle of the body? Nothing saved.")
        # The echo read `found`; it should read `expected`. Path = direct + extra, so move the direct path by the difference.
        direct = geometry.direct_m[k] + expected[k] - float(np.mean(found))
        new.append(direct)
        print(f"  {name:<12} direct path {geometry.direct_m[k] * 100:5.1f} cm -> {direct * 100:5.1f} cm "
              f"(by ruler: {np.linalg.norm(geometry.mics[k] - geometry.speaker) * 100:.1f} cm)")
    saved = json.loads(ears_module.CALIBRATION_FILE.read_text()) if ears_module.CALIBRATION_FILE.exists() else {}
    saved[geometry.mode] = {"direct_m": [round(v, 4) for v in new], "wall_m": wall, "when": time.strftime("%Y-%m-%d %H:%M")}
    ears_module.CALIBRATION_FILE.write_text(json.dumps(saved, indent=2) + "\n")
    print(f"  saved to {ears_module.CALIBRATION_FILE}. Run again without --wall: the wall should read {wall:.2f} m.")


class StillDog:
    """A dog that is somewhere and does not move (for --sim)."""

    def __init__(self, x: float, y: float, heading_deg: float):
        self.x, self.y, self.heading = x, y, np.radians(heading_deg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the microphones and hear the first echoes.")
    parser.add_argument("--list", action="store_true", help="list the sound devices and stop")
    parser.add_argument("--tap", action="store_true", help="which channel is which mic")
    parser.add_argument("--ears", choices=("two", "four"), default="two")
    parser.add_argument("--pings", type=int, default=5)
    parser.add_argument("--wall", type=float, default=None, metavar="D", help="calibrate against a wall D m dead ahead")
    parser.add_argument("--mic", default="DJI")
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--save", default=None, metavar="FILE", help="also save the recordings (numpy .npy)")
    parser.add_argument("--sim", action="store_true", help="the pretend room: 1.0 m from the wall ahead")
    args = parser.parse_args(argv)

    if args.list:
        from sound import list_devices
        print(list_devices())
        return 0
    geometry = ears_module.load_geometry(args.ears)
    if args.sim:
        from sound import SimSound
        room = SimRoom.box(4.0, 3.0, seed=2)
        sound = SimSound(room, StillDog(3.0, 1.3, 0.0), geometry.speaker, geometry.mics)   # the wall x = 4 is 1.0 m ahead
    else:
        from sound import SoundCard
        sound = SoundCard(len(geometry.mics), mic=args.mic, speaker=args.speaker)
    if args.tap:
        tap_test(sound)
        return 0
    return 0 if report(geometry, sound, args.pings, args.wall, args.save) else 1


if __name__ == "__main__":
    sys.exit(main())
