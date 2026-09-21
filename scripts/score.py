"""Procedural score + SFX for the blank-poem video. Pure DSP, no models.

Everything is synthesised from the timeline handed in by the caller, so the
music lands on the same beats as the picture.
"""
import math
import numpy as np
from scipy import signal

SR = 48000


def _t(dur):
    return np.arange(int(round(dur * SR))) / SR


def _env_ad(n, attack, decay):
    e = np.ones(n)
    a = max(1, int(attack * SR))
    e[:a] = np.linspace(0, 1, a) ** 1.5
    t = np.arange(n) / SR
    e *= np.exp(-t / max(1e-4, decay))
    k = max(1, int(0.004 * SR))
    e[-k:] *= np.linspace(1, 0, k)
    return e


def bell(freq, dur, amp, rng, decay=1.35):
    """Soft music-box / felt-piano tone."""
    n = int(round(dur * SR))
    t = _t(dur)
    sig = np.zeros(n)
    for mult, a, d in ((1.0, 1.0, 1.0), (2.0, 0.26, 1.7), (3.01, 0.10, 2.4), (4.98, 0.03, 3.4)):
        sig += a * np.sin(2 * np.pi * freq * mult * t + 0.6 * mult) * np.exp(-t / (decay / d))
        sig += 0.022 * np.exp(-t / 0.004) * rng.standard_normal(n) * mult ** -1.5
    sig *= _env_ad(n, 0.003, decay)
    b, a = signal.butter(2, 9000 / (SR / 2), btype="low")
    return amp * signal.filtfilt(b, a, sig)


def pad(freqs, dur, amp, detune=0.0035):
    """Warm sustained chord with a slow swell."""
    n = int(round(dur * SR))
    t = _t(dur)
    swell = np.minimum(1.0, t / 0.9) * np.minimum(1.0, np.maximum(0.0, (dur - t) / 1.2))
    sig = np.zeros(n)
    for i, f in enumerate(freqs):
        for k, (df, w) in enumerate(((0.0, 1.0), (detune, 0.55), (-detune * 0.8, 0.45))):
            sig += w * np.sin(2 * np.pi * f * (1 + df) * t + 1.3 * i + 0.7 * k)
        sig += 0.10 * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t / (dur * 0.5))
    return sig * swell * amp / (len(freqs) * 2.0)


def _place(dst, sig, t0, pan=0.0):
    i0 = int(round(t0 * SR))
    if i0 >= len(dst) or i0 + len(sig) < 0:
        return
    a, b = max(0, i0), min(len(dst), i0 + len(sig))
    seg = sig[a - i0:b - i0]
    dst[a:b, 0] += seg * math.sqrt(0.5 * (1 - pan))
    dst[a:b, 1] += seg * math.sqrt(0.5 * (1 + pan))


def _noise(dur, lo, hi, amp, attack, decay, seed):
    n = int(dur * SR)
    x = np.random.default_rng(seed).standard_normal(n)
    b, a = signal.butter(2, [lo / (SR / 2), min(0.98, hi / (SR / 2))], btype="band")
    return amp * signal.filtfilt(b, a, x) * _env_ad(n, attack, decay)


def _whoosh(dur, amp, seed):
    """Noise swept dark -> airy: the sound of a piece travelling."""
    n = int(dur * SR)
    x = np.random.default_rng(seed).standard_normal(n)
    cut = np.linspace(260, 1450, n)
    y = np.zeros(n)
    lp = 0.0
    for i in range(n):
        k = 1 - math.exp(-2 * math.pi * cut[i] / SR)
        lp += k * (x[i] - lp)
        y[i] = lp
    for btype, cut_hz in (("high", 240), ("low", 2100)):
        b, a = signal.butter(2, cut_hz / (SR / 2), btype=btype)
        y = signal.filtfilt(b, a, y)
    t = np.arange(n) / SR
    return amp * y * np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.4


def _reverb(x, decay=0.62, length=1.5, wet=0.28):
    n = int(length * SR)
    t = _t(length)
    out = np.zeros_like(x)
    for ch in range(2):
        ir = np.random.default_rng(100 + ch).standard_normal(n) * np.exp(-t / decay)
        b, a = signal.butter(2, 3000 / (SR / 2), btype="low")
        ir = signal.filtfilt(b, a, ir)
        ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
        out[:, ch] = signal.fftconvolve(x[:, ch], ir)[:len(x)] * wet
    return out


# Fmaj7 -> Am7 -> Cmaj7 -> Bbmaj7 -> Fmaj7, warm and unresolved enough to loop
_CHORDS = [
    [174.61, 220.00, 261.63, 329.63],
    [220.00, 261.63, 329.63, 392.00],
    [130.81, 164.81, 196.00, 246.94],
    [233.08, 293.66, 349.23, 220.00],
    [174.61, 220.00, 261.63, 349.23],
]
_PENTATONIC = [440.00, 523.25, 587.33, 659.25, 783.99, 880.00, 1046.50, 1174.66]


def melody_for(lands, base_index=0):
    """One rising pentatonic note per landed slice: the poem itself plays the tune."""
    return [(t, _PENTATONIC[min(len(_PENTATONIC) - 1, base_index + i)]) for i, t in enumerate(lands)]


def build_mix(duration, picks, lands, flights, seed=7, melody=None, sfx=True,
              return_stems=False):
    rng = np.random.default_rng(seed)
    n = int(round(duration * SR))
    song = np.zeros((n, 2))
    effects = np.zeros((n, 2))

    chord_dur = max(1.6, duration / 5.2)
    for idx, freqs in enumerate(_CHORDS):
        t0 = idx * chord_dur
        if t0 > duration:
            break
        _place(song, pad(freqs, min(chord_dur + 0.7, duration - t0 + 0.6), 0.34), t0)

    for f, t0 in (melody or []):
        _place(song, bell(f, 2.8, 0.185, rng, decay=1.5), t0 + 0.02)
        _place(song, bell(f / 2, 2.6, 0.07, rng, decay=1.6), t0 + 0.02, pan=-0.25)
    if duration > 2.0:
        _place(song, bell(329.63, 2.6, 0.16, rng), 0.35)
        _place(song, bell(392.00, 2.6, 0.13, rng), 0.80, pan=0.3)
    if duration > 3.4:
        _place(song, bell(523.25, 3.2, 0.17, rng), duration - 2.35, pan=0.15)
        _place(song, bell(659.25, 3.4, 0.13, rng), duration - 1.75, pan=-0.2)

    air = np.random.default_rng(seed + 5).standard_normal(n) * 0.006
    b, a = signal.butter(2, 900 / (SR / 2), btype="low")
    air = signal.filtfilt(b, a, air)
    ramp = np.minimum(1.0, np.arange(n) / (0.8 * SR)) * \
        np.minimum(1.0, np.maximum(0.0, (n - np.arange(n)) / (1.2 * SR)))
    air *= ramp
    song[:, 0] += air
    song[:, 1] += np.roll(air, 197)

    if sfx:
        for i, t in enumerate(picks):
            _place(effects, _noise(0.09, 700, 2600, 0.155, 0.001, 0.022, 10 + i), t)
            _place(effects, bell(1180 - 60 * i, 0.10, 0.095, rng), t)
        for i, t in enumerate(lands):
            _place(effects, _noise(0.05, 350, 1800, 0.135, 0.001, 0.012, 60 + i), t)
            tap = (np.sin(2 * np.pi * 196 * _t(0.16)) * _env_ad(int(0.16 * SR), 0.002, 0.035)
                   + 0.4 * np.sin(2 * np.pi * 392 * _t(0.16)) * _env_ad(int(0.16 * SR), 0.002, 0.02))
            _place(effects, 0.150 * tap, t)
        for i, fl in enumerate(flights):
            w = _whoosh(fl["t1"] - fl["t0"] + 0.06, 0.160, 30 + i)
            steps = 40
            for s in range(steps):
                a0 = int(round(s * len(w) / steps))
                a1 = int(round((s + 1) * len(w) / steps))
                pan = fl["pan"](s / steps)
                _place(effects, w[a0:a1], fl["t0"] + s * (fl["t1"] - fl["t0"]) / steps, pan=pan)

    song = song + _reverb(song) * 0.9
    effects = effects + _reverb(effects, decay=0.35, length=0.8, wet=0.12) * 0.8
    stems = {"music": song, "sfx": effects}
    mix = song + effects

    b, a = signal.butter(2, 55 / (SR / 2), btype="high")
    mix = signal.filtfilt(b, a, mix, axis=0)
    peak = float(np.max(np.abs(mix))) + 1e-9
    mix = np.tanh(mix / peak * 1.15) / np.tanh(1.15)
    fade_in = int(0.45 * SR)
    mix[:fade_in] *= np.linspace(0, 1, fade_in)[:, None] ** 1.4
    fade_out = int(min(1.1, duration * 0.2) * SR)
    mix[-fade_out:] *= np.linspace(1, 0, fade_out)[:, None] ** 1.6
    mix = mix * (0.89 / (float(np.max(np.abs(mix))) + 1e-9))
    if return_stems:
        return {"mix": mix, "music": song, "sfx": effects}
    return mix
