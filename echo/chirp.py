"""The sound Pupper makes, and how to find its echoes in what the ears heard.

The one big idea
----------------
Sound travels 343 m/s, so time is distance: an echo that arrives 6 ms after
the sound left has travelled 2 m, there and back from a wall 1 m away. To
measure a time we need a sound with a sharp "now" in it. A click has one but
is quiet. A *chirp* (a tone sweeping from CHIRP_F0 up to CHIRP_F1) is long
and loud, and still has a sharp now: slide a copy of the chirp along the
recording and it lines up with itself at exactly one instant. That sliding
comparison is `compress` (radar and bats call it pulse compression); it
squeezes the 8 ms chirp into a spike about 1/bandwidth = 0.1 ms wide, which
is 3.4 cm of travel.

We never need to know when the sound left the speaker
----------------------------------------------------
The speaker and the microphones are different sound cards (and the DJI mics
are a radio link with its own delay), so "when did we play it" is unknown by
tens of milliseconds: metres of error. But every ear hears the chirp twice:
first straight from the speaker a few centimetres away (the *direct* sound,
always the loudest), then from the walls. We time each echo from the direct
sound in the *same ear*:

    path of the echo = (speaker -> ear distance, by ruler) + 343 * (echo time - direct time)

Every delay in the chain is in both arrivals and cancels. The ears do not
even have to be in step with each other. This is why the ears ride on the dog.

A ping is CHIRPS_PER_PING chirps. The profile of each is lined up on its
direct sound and averaged, so noise that is not locked to the chirp fades.
The gaps between the chirps are uneven on purpose: the room is still ringing
from the previous chirp when the next one goes out, and with uneven gaps that
ringing lands in a different place each time and fades like noise too.

Numpy only: this runs on the Pupper and on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- the physics ----------------------------------------------------------------
SAMPLE_RATE = 48000        # Hz: what the DJI Mic 3 sends, and what we play
SPEED_OF_SOUND = 343.0     # m/s at 20 C (it changes 0.17 % per degree: 2 mm per metre)

# --- the chirp ------------------------------------------------------------------
CHIRP_F0 = 3000.0          # Hz. Low end: a small speaker is weak below this, motors are loud below this
CHIRP_F1 = 13000.0         # Hz. High end: the DJI link carries up to 20 kHz; soft things absorb the top
CHIRP_S = 0.008            # s: longer is louder (more energy), not blurrier
CHIRP_FADE = 0.25          # fraction of the chirp spent fading in and out (no click at the ends)
VOLUME = 0.8               # of full scale

# --- a ping: a few chirps --------------------------------------------------------
LEAD_S = 0.05                              # silence before the first chirp
CHIRP_STARTS_S = (0.0, 0.085, 0.160, 0.250)  # when each chirp starts, after the lead: uneven gaps
TAIL_S = 0.08                              # keep recording this long after the last chirp starts
CHIRPS_PER_PING = len(CHIRP_STARTS_S)

# --- the profile -----------------------------------------------------------------
PROFILE_BEFORE_M = 0.5     # m of path kept before the direct sound (to measure the noise)
PROFILE_AFTER_M = 8.0      # m of extra path kept after it: a wall 4 m away


def make_chirp() -> np.ndarray:
    """One chirp: a sine sweeping CHIRP_F0 -> CHIRP_F1 in CHIRP_S, faded in and out."""
    n = int(round(CHIRP_S * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    # The phase of a sweep is the integral of its frequency: f(t) = F0 + (F1 - F0) * t / T.
    phase = 2 * np.pi * (CHIRP_F0 * t + 0.5 * (CHIRP_F1 - CHIRP_F0) * t * t / CHIRP_S)
    fade = int(n * CHIRP_FADE / 2)
    window = np.ones(n)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
    window[:fade], window[n - fade:] = ramp, ramp[::-1]
    return VOLUME * window * np.sin(phase)


def make_ping() -> np.ndarray:
    """What the speaker plays for one ping: silence, the chirps, silence."""
    chirp = make_chirp()
    total = int(round((LEAD_S + CHIRP_STARTS_S[-1] + TAIL_S) * SAMPLE_RATE))
    ping = np.zeros(total)
    for start in CHIRP_STARTS_S:
        i = int(round((LEAD_S + start) * SAMPLE_RATE))
        ping[i:i + len(chirp)] += chirp
    return ping


def band_window(freqs: np.ndarray) -> np.ndarray:
    """A smooth hill over the chirp's band, zero outside it.

    Compressing with a flat band gives a spike with ripples beside it that
    fall off slowly; next to the direct sound those ripples would bury the
    echo of a near wall. A smooth band trades a slightly wider spike for
    ripples that are gone within a few centimetres.
    """
    inside = (freqs >= CHIRP_F0) & (freqs <= CHIRP_F1)
    hill = 0.5 - 0.5 * np.cos(2 * np.pi * (freqs - CHIRP_F0) / (CHIRP_F1 - CHIRP_F0))
    return np.where(inside, hill, 0.0)


def compress(recording: np.ndarray) -> np.ndarray:
    """Slide the chirp along one ear's recording: a spike wherever a chirp arrives.

    recording:  1-D array, one ear.
    Returns:    complex array, same length. Its size (abs) is how well the
                chirp lines up at each sample; a chirp that *starts* at sample
                k puts its spike at k. Complex, so that chirps can be averaged
                before taking the size (noise then cancels instead of adding).

    In the frequency domain, sliding-and-comparing is a multiplication by the
    chirp's conjugate spectrum. We also divide by the chirp's own strength at
    each frequency and shape the band (band_window), and keep only positive
    frequencies, which makes the result complex (its abs is the envelope).
    """
    n = len(recording)
    chirp = make_chirp()
    size = 1 << int(np.ceil(np.log2(n + len(chirp))))
    freqs = np.fft.fftfreq(size, 1.0 / SAMPLE_RATE)
    chirp_spectrum = np.fft.fft(chirp, size)
    power = np.abs(chirp_spectrum) ** 2
    matched = np.conj(chirp_spectrum) / (power + 1e-3 * power.max()) * band_window(freqs)
    return (2.0 * np.fft.ifft(np.fft.fft(recording, size) * matched))[:n]


def shift_samples(signal: np.ndarray, shift: float) -> np.ndarray:
    """Delay a complex signal by a (possibly fractional) number of samples."""
    freqs = np.fft.fftfreq(len(signal))
    return np.fft.ifft(np.fft.fft(signal) * np.exp(-2j * np.pi * freqs * shift))


def peak_position(size: np.ndarray, i: int) -> float:
    """Where the peak near sample i really is, between samples (a parabola through three points)."""
    if i <= 0 or i >= len(size) - 1:
        return float(i)
    a, b, c = size[i - 1], size[i], size[i + 1]
    bend = a - 2 * b + c
    return float(i) if bend == 0 else i + 0.5 * (a - c) / bend


@dataclass
class Profile:
    """What one ear heard, lined up on the direct sound.

    extra_m[k] is extra path in metres: 0 is the direct sound, 2.0 is sound
    that travelled 2 m further than the direct sound did. size[k] is how
    strong the arrival there is, with the direct sound = 1.
    """

    extra_m: np.ndarray       # the axis, PROFILE_BEFORE_M before the direct sound to PROFILE_AFTER_M after
    size: np.ndarray          # arrival strength at each extra_m (direct sound = 1)
    direct_level: float       # how loud the direct sound was, in recording units (1.0 = full scale chirp)
    direct_at_s: float        # when the first chirp's direct sound arrived in the recording
    noise: float              # typical size before the direct sound arrived
    steadiness: float         # 1.0 = every chirp's direct sound was equally loud; low = auto gain, dropouts
    clipped: bool             # the recording hit full scale


def echo_profile(recording: np.ndarray) -> Profile:
    """One ear's recording of a ping -> its Profile.

    1. compress: spikes where chirps arrive.
    2. find the direct sound of every chirp: the chirps were played at known
       gaps (CHIRP_STARTS_S), so slide that pattern along and take the best fit.
    3. cut out the stretch around each direct sound, line the stretches up to
       a fraction of a sample, and average them.
    """
    spikes = compress(np.asarray(recording, dtype=float))
    size = np.abs(spikes)
    before = int(round(PROFILE_BEFORE_M / SPEED_OF_SOUND * SAMPLE_RATE))
    after = int(round(PROFILE_AFTER_M / SPEED_OF_SOUND * SAMPLE_RATE))
    offsets = [int(round(s * SAMPLE_RATE)) for s in CHIRP_STARTS_S]

    # Step 2: add up the size at the four places the direct sounds would be, for every start.
    last = len(size) - offsets[-1] - after
    if last <= before:
        raise ValueError("recording too short for a ping")
    fit = sum(size[before + o:last + o] for o in offsets)
    first = before + int(np.argmax(fit))

    # Step 3.
    pieces, levels = [], []
    for o in offsets:
        near = first + o
        i = near - 8 + int(np.argmax(size[near - 8:near + 9]))  # the two sound cards' clocks differ a little
        exact = peak_position(size, i)
        piece = spikes[i - before:i + after + 1]
        pieces.append(shift_samples(piece, i - exact))
        levels.append(size[i])
    mean = np.mean(pieces, axis=0)
    level = float(np.abs(mean[before]))
    profile = np.abs(mean) / max(level, 1e-12)
    quiet = profile[:before - int(0.1 / SPEED_OF_SOUND * SAMPLE_RATE)]
    return Profile(
        extra_m=(np.arange(len(mean)) - before) * SPEED_OF_SOUND / SAMPLE_RATE,
        size=profile,
        direct_level=level,
        direct_at_s=(first) / SAMPLE_RATE,
        noise=float(np.median(quiet)),
        steadiness=float(min(levels) / max(max(levels), 1e-12)),
        clipped=bool(np.max(np.abs(recording)) >= 0.999),
    )
