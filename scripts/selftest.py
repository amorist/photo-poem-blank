#!/usr/bin/env python3
"""End-to-end smoke test: many photo shapes, sizes and formats -> video + score.

    python3 selftest.py            # fast synthetic matrix
    python3 selftest.py --full     # same matrix with the shipped timeline (slower)

Every case must produce an mp4 with a real audio track: music present (not silent),
SFX landing on the cut/land beats, and loudness at the target.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import poem_video as P  # noqa: E402

POEM = ["这一行放着{0}和{1}，", "第二行{2}与{3}并排，", "最后一行只剩下{4}和一点安静。"]


def make_photo(path, size, kind="colour"):
    """A synthetic but photo-like image: sky gradient, ground, distinct objects."""
    w, h = size
    im = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(im)
    for y in range(h):
        t = y / max(1, h - 1)
        d.line([(0, y), (w, y)], fill=(int(120 + 90 * t), int(170 + 70 * t), int(230 - 30 * t)))
    horizon = int(h * 0.55)
    d.rectangle([0, horizon, w, h], fill=(96, 122, 84))
    d.ellipse([w * 0.08, horizon - h * 0.30, w * 0.30, horizon - h * 0.06], fill=(250, 244, 210))
    d.rectangle([w * 0.55, h * 0.58, w * 0.70, h * 0.80], fill=(198, 74, 62))
    d.ellipse([w * 0.72, h * 0.10, w * 0.94, h * 0.34], fill=(255, 255, 255))
    d.rectangle([w * 0.12, h * 0.78, w * 0.42, h * 0.94], fill=(38, 42, 48))
    for i in range(6):
        x = w * (0.15 + 0.13 * i)
        d.line([(x, horizon), (x, h)], fill=(70, 92, 62), width=max(2, w // 300))
    if kind == "grey":
        im = im.convert("L").convert("RGB")
    elif kind == "alpha":
        im = im.convert("RGBA")
        mask = Image.new("L", (w, h), 255)
        md = ImageDraw.Draw(mask)
        md.ellipse([-w * 0.2, -h * 0.2, w * 0.5, h * 0.5], fill=0)
        md.ellipse([w * 0.6, h * 0.7, w * 1.2, h * 1.2], fill=0)
        im.putalpha(mask)
    im.save(path)
    return path


def auto_config(image_path, out_dir, basename, full=False):
    """Build a config: auto photo window, five spread-out slices, generic poem."""
    tl = {} if full else {"slice_gap": 1.02, "tail": 0.60}
    n_slots = 5
    probe = {"image": image_path, "lines": POEM,
             "slices": [{"key": f"probe{i}", "box": [0, 0, 20, 14]} for i in range(n_slots)],
             "timeline": tl}
    tmp_cfg = os.path.join(out_dir, "_probe.json")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(probe, open(tmp_cfg, "w"))
    poem = P.Poem(P.load_config(tmp_cfg))
    x0, y0, x1, y1 = poem.box
    scale = poem.scale

    want_w, want_h = 170 / scale, 106 / scale          # canvas-sized markers -> source px
    margin = 20 / scale                                # keep 20 canvas px clear of the edges
    slots = [(260, 880), (540, 880), (820, 880), (260, 1200), (540, 1200)]
    slices = []
    for i, (cx, cy) in enumerate(slots):
        sx = min(max(x0 + (cx - want_w / 2) / scale, x0 + margin), x1 - want_w - margin)
        sy = min(max(y0 + (cy - poem.split - want_h / 2) / scale, y0 + margin),
                 y1 - want_h - margin)
        slices.append({"key": f"s{i}", "box": [int(round(sx)), int(round(sy)),
                                              int(round(want_w)), int(round(want_h))]})
    cfg = {"image": image_path, "out_dir": out_dir, "basename": basename,
           "lines": POEM, "slices": slices, "timeline": tl}
    path = os.path.join(out_dir, "config.json")
    json.dump(cfg, open(path, "w"), ensure_ascii=False, indent=2)
    os.remove(tmp_cfg)
    return path


def pcm(mp4, sr=24000):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", mp4, "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, dtype="<f4"), sr


def tonal_ratio(x):
    """Peak-to-median energy ratio of the spectrum: high for music, ~1 for noise."""
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    return float(np.percentile(spec, 99.9) / (np.median(spec) + 1e-12))


def event_hits(stem, sr, times, window=0.14):
    """How many expected event times sit on a local energy spike of the SFX stem."""
    mono = stem.mean(axis=1)
    energy = np.convolve(mono ** 2, np.ones(int(0.02 * sr)) / (0.02 * sr), mode="same")
    hits = 0
    for t in times:
        i = int(t * sr)
        lo, hi = max(0, i - int(window * sr)), min(len(energy), i + int(window * sr))
        if hi > lo and energy[lo:hi].max() > 6 * (np.percentile(energy, 75) + 1e-12):
            hits += 1
    return hits


def best_corr(a, b, sr, max_lag=0.25):
    """Best normalised correlation between two signals over a small lag search."""
    n = min(len(a), len(b))
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    best = (0.0, 0)
    for lag in range(-int(max_lag * sr), int(max_lag * sr) + 1, 4):
        if lag >= 0:
            x, y = a[lag:n], b[:n - lag]
        else:
            x, y = a[:n + lag], b[-lag:n]
        if len(x) < sr // 2:
            continue
        denom = math.sqrt(float((x ** 2).sum()) * float((y ** 2).sum())) + 1e-12
        c = float((x * y).sum()) / denom
        if abs(c) > abs(best[0]):
            best = (c, lag)
    return best


def check(out_dir, basename, cfg_path, stdout, source_size):
    problems = []
    mp4 = os.path.join(out_dir, f"{basename}-1080x1440.mp4")
    nocover = os.path.join(out_dir, f"{basename}-nocover-1080x1440.mp4")
    cover = os.path.join(out_dir, f"{basename}-cover-1080x1440.jpg")
    for f in (mp4, nocover, cover):
        if not os.path.exists(f):
            problems.append(f"missing {os.path.basename(f)}")
    if problems:
        return problems, {}
    if "checks: ok" not in stdout:
        problems.append("layout QA reported warnings")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "stream=codec_type,codec_name,width,height,sample_rate,channels:"
                            "format=duration", "-of", "json", mp4],
                           capture_output=True, text=True).stdout
    info = json.loads(probe)
    streams = {s["codec_type"]: s for s in info["streams"]}
    if "video" not in streams or "audio" not in streams:
        problems.append("missing video or audio stream")
        return problems, {}
    v, a = streams["video"], streams["audio"]
    if (v["width"], v["height"]) != (1080, 1440):
        problems.append(f"canvas is {v['width']}x{v['height']}")
    if a.get("sample_rate") != "48000" or a.get("channels") != 2:
        problems.append(f"audio is {a.get('sample_rate')}Hz x{a.get('channels')}")
    dur = float(info["format"]["duration"])
    lufs = P.measure_lufs(mp4)
    if abs(lufs - (-14.0)) > 0.8:
        problems.append(f"loudness {lufs} LUFS")
    x, sr = pcm(mp4)
    rms = float(np.sqrt((x ** 2).mean())) if len(x) else 0.0
    if rms < 0.03:
        problems.append(f"audio too quiet (rms {rms:.3f})")

    # the score itself: music stem must be tonal and audible, SFX stem must land on the beats
    poem = P.Poem(P.load_config(cfg_path))
    stems = P.score_signals(poem, dur, poem.cfg, sfx=True, stems=True)
    music, sfx_stem, mix = stems["music"], stems["sfx"], stems["mix"]
    music_rms = float(np.sqrt((music ** 2).mean()))
    sfx_rms = float(np.sqrt((sfx_stem ** 2).mean()))
    tone = tonal_ratio(music.mean(axis=1))
    if music_rms < 0.03:
        problems.append(f"music stem too quiet ({music_rms:.3f})")
    if tone < 30:
        problems.append(f"music stem does not look tonal (peak/median {tone:.1f})")
    if not 0.02 < sfx_rms / max(music_rms, 1e-9) < 0.8:
        problems.append(f"sfx/music balance off ({20 * math.log10(sfx_rms / music_rms):.1f} dB)")

    off = poem.offset()
    events = []
    for i in poem.order():
        t0, pick_end, land, _ = poem.timeline()[i]
        events += [t0 - off, pick_end - off, land - off]
    hits = event_hits(sfx_stem, score_sr(), events)
    need = max(4, int(0.6 * len(poem.order()) * 2))
    if hits < need:
        problems.append(f"SFX land on only {hits}/{len(events)} beats")

    # the muxed audio must actually be that score
    mix24 = np.interp(np.arange(0, len(mix), 2.0), np.arange(len(mix)), mix.mean(axis=1))
    corr, lag = best_corr(x, mix24, sr)
    if corr < 0.8:
        problems.append(f"delivered audio does not match the score (r={corr:.2f})")
    return problems, dict(dur=dur, lufs=lufs, rms=rms, music=music_rms, sfx=sfx_rms,
                          tone=tone, hits=hits, beats=len(events), corr=corr,
                          src=f"{source_size[0]}x{source_size[1]}")


def score_sr():
    return P._score().SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="use the shipped timeline (slower)")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    root = tempfile.mkdtemp(prefix="photo-poem-selftest-")
    src_dir = os.path.join(root, "sources")
    os.makedirs(src_dir)
    cases = [
        ("landscape-16x9", (1600, 900), "jpg", "colour"),
        ("portrait-2x3", (900, 1350), "jpg", "colour"),
        ("square", (1000, 1000), "png", "colour"),
        ("panorama-3x1", (2400, 800), "jpg", "colour"),
        ("greyscale", (1200, 800), "jpg", "grey"),
        ("alpha-png", (1000, 1200), "png", "alpha"),
        ("tiny-480x320", (480, 320), "jpg", "colour"),
        ("huge-3600x2400", (3600, 2400), "jpg", "colour"),
    ]
    if a.only:
        cases = [c for c in cases if c[0] in a.only.split(",")]

    rows, failed = [], 0
    for name, size, ext, kind in cases:
        path = os.path.join(src_dir, f"{name}.{ext}")
        make_photo(path, size, kind)
        out_dir = os.path.join(root, name)
        cfg = auto_config(path, out_dir, f"t-{name}", a.full)
        p = subprocess.run([sys.executable, os.path.join(HERE, "poem_video.py"), "build", cfg],
                           capture_output=True, text=True)
        problems, stats = check(out_dir, f"t-{name}", cfg, p.stdout, size)
        rows.append((name, size, stats, problems))
        if problems:
            failed += 1
            print(f"  FAIL {name}: {problems}\n{p.stdout[-600:]}{p.stderr[-600:]}")
        else:
            print(f"  ok   {name:16s} {stats['src']:>10s}  {stats['dur']:5.2f}s  "
                  f"{stats['lufs']:6.1f}LUFS  music {stats['music']:.3f} / sfx {stats['sfx']:.3f}"
                  f"  tone {stats['tone']:.0f}  SFX beats {stats['hits']}/{stats['beats']}"
                  f"  audio-match r={stats['corr']:.3f}")

    extra = []
    exif_path = os.path.join(src_dir, "exif-rotated.jpg")
    make_photo(os.path.join(src_dir, "_raw.jpg"), (1500, 1000))
    raw = Image.open(os.path.join(src_dir, "_raw.jpg"))
    exif = Image.Exif()
    exif[274] = 6                                  # orientation: rotate 90° CW
    raw.save(exif_path, exif=exif)
    got = P.open_image(exif_path).size
    if got != (1000, 1500):
        extra.append(f"EXIF rotation not applied (got {got}, want (1000, 1500))")

    heic = os.path.join(src_dir, "iphone.heic")
    if shutil.which("sips"):
        subprocess.run(["sips", "-s", "format", "heic", os.path.join(src_dir, "_raw.jpg"),
                        "--out", heic], capture_output=True)
    if os.path.exists(heic):
        try:
            P.open_image(heic)
            print("  ok   heic-fallback    (opened through the sips/ffmpeg fallback)")
        except SystemExit as e:
            extra.append(f"HEIC fallback failed: {e}")
    else:
        print("  skip heic-fallback    (sips cannot encode HEIC here)")
    rows_extra = extra
    for e in rows_extra:
        print("  FAIL format:", e)

    print("=" * 72)
    print(f"{len(rows) - failed}/{len(rows)} photo cases passed"
          + ("" if not rows_extra else f"; {len(rows_extra)} format check(s) failed"))
    print("artifacts:", root if a.keep else "(temporary, removed)")
    if not a.keep:
        shutil.rmtree(root, ignore_errors=True)
    sys.exit(1 if (failed or rows_extra) else 0)


if __name__ == "__main__":
    main()
