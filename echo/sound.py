"""Make the ping and record what comes back: the real sound card, or the pretend room.

Both have one method, which is all navigator.py and mic_check.py use:

    sound.ping()  ->  (samples, ears) array: what each ear recorded while the ping played

SoundCard plays the ping on Pupper's speaker and records the DJI receiver at
the same time. The two are different devices with different clocks, and the
mics are a radio link, so the recording starts at an unknown moment relative
to the ping. That is fine: chirp.py times every echo from the direct sound in
the same ear, so we only need the recording to *contain* the whole ping
(EXTRA_S of slack at the end).

On the receiver (once, in its menu or the DJI Mimo app). These all change the
sound in ways that hide echoes:

    Channel mode      Stereo (two mics) or Quadraphonic (four): one mic per channel
    Noise cancelling  OFF   (it is built to remove exactly what we listen for)
    Auto / adaptive gain  OFF, a fixed gain instead
    Low cut           OFF or on, either (we use 3 kHz and up)
    Voice tone        Regular
    Lossless audio    ON if it has it

sounddevice (PortAudio) is the only thing here that is not already on the
Pupper:  sudo apt install libportaudio2 && pip install --user --break-system-packages sounddevice
"""

from __future__ import annotations

import time

import numpy as np

import chirp

EXTRA_S = 0.35   # keep recording this long after the ping has been handed to the speaker


def list_devices() -> str:
    """Every sound device, with how many channels it records and plays."""
    import sounddevice as sd

    lines = []
    for i, d in enumerate(sd.query_devices()):
        lines.append(f"  {i:2d}  in {d['max_input_channels']}  out {d['max_output_channels']}  "
                     f"{int(d['default_samplerate'])} Hz  {d['name']}")
    return "\n".join(lines)


def find_device(name: str | int | None, kind: str) -> int | None:
    """A device number from a number, part of a name ("DJI", "Wireless"), or None for the system default."""
    import sounddevice as sd

    if name is None:
        return None
    if isinstance(name, int) or str(name).isdigit():
        return int(name)
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, d in enumerate(sd.query_devices()):
        if str(name).lower() in d["name"].lower() and d[key] > 0:
            return i
    raise ValueError(f"no {kind} device with '{name}' in its name. The devices are:\n{list_devices()}")


class SoundCard:
    """Pupper's speaker and the DJI receiver."""

    def __init__(self, ears: int, mic: str | int | None = "DJI", speaker: str | int | None = None):
        """ears: how many channels to record. mic, speaker: a device number or part
        of its name; None is the system's default device."""
        import sounddevice as sd

        self.sd = sd
        self.ears = ears
        self.mic = find_device(mic, "input")
        self.speaker = find_device(speaker, "output")
        has = sd.query_devices(self.mic, "input")
        if has["max_input_channels"] < ears:
            raise ValueError(
                f"'{has['name']}' records {has['max_input_channels']} channel(s), this mode needs {ears}. "
                "Two mics: set the receiver to Stereo. Four: Quadraphonic, on the main receiver. "
                "If the computer still sees too few channels, use the mode with fewer ears.")
        self.speaker_channels = min(2, sd.query_devices(self.speaker, "output")["max_output_channels"])
        self.name = f"{has['name']} -> {sd.query_devices(self.speaker, 'output')['name']}"

    def ping(self) -> np.ndarray:
        """Play one ping and return what the ears recorded: (samples, ears), -1..1."""
        sd = self.sd
        sound = chirp.make_ping().astype(np.float32)
        out = np.ascontiguousarray(np.repeat(sound[:, None], self.speaker_channels, axis=1))
        blocks: list[np.ndarray] = []

        def on_audio(indata, _frames, _time, _status) -> None:
            blocks.append(indata.copy())

        with sd.InputStream(samplerate=chirp.SAMPLE_RATE, device=self.mic, channels=self.ears,
                            dtype="float32", callback=on_audio):
            with sd.OutputStream(samplerate=chirp.SAMPLE_RATE, device=self.speaker,
                                 channels=self.speaker_channels, dtype="float32") as speaker:
                speaker.write(out)       # returns once the sound is handed over, not once it has played
                time.sleep(EXTRA_S)
        return np.concatenate(blocks)[:, :self.ears].astype(float)


class SimSound:
    """The pretend room (sim_room.py), heard from wherever the pretend dog really is."""

    name = "simulated room"

    def __init__(self, room, dog, speaker: np.ndarray, mics: np.ndarray):
        """speaker, mics: where they *really* are on the pretend dog (the tests
        move them a little from what ears.yaml says, like a ruler would)."""
        self.room, self.dog, self.speaker, self.mics = room, dog, speaker, mics

    def ping(self) -> np.ndarray:
        return self.room.record((self.dog.x, self.dog.y, self.dog.heading), self.speaker, self.mics)
