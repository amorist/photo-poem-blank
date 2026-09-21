#!/usr/bin/env python3
"""Turn one photo into a vertical "poem with cut-out blanks" video.

The mechanic: a short Chinese poem sits on a warm paper panel; several concrete
details are lifted out of the photo below and fly into the poem's blanks, each
leaving a translucent marker where it came from. Deterministic and offline:
Pillow for frames, ffmpeg for encoding, pure DSP for the score.

    python3 poem_video.py build config.json
    python3 poem_video.py stills config.json --times 2.6,9.6
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave

from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _score():
    """Imported lazily: the score pulls in numpy+scipy (~0.7s) which the layout
    and the crop helper do not need."""
    import score
    return score

DEFAULTS = {
    "image": None,
    "out_dir": None,
    "basename": "photo-poem",
    "canvas": {"w": 1080, "h": 1440},
    "fps": 30,
    "split": 0.48,                       # photo occupies the lower 52%
    "bg": [241, 241, 239],               # #F1F1EF
    "ink": [17, 17, 17],
    "font": {"path": None, "index": 0, "size_ratio": 0.032, "line_height": 1.88},
    "slice": {"height_ratio": 1.6, "radius": 3, "gap": 5},
    "marker": {"fill_alpha": 95, "edge_alpha": 200, "edge_width": 2},
    "timeline": {"cover_hold": 0.80, "dissolve": 0.25, "slice_start": 2.10,
                 "slice_gap": 1.12, "pick": 0.14, "wipe": 0.12, "fly": 0.62,
                 "settle": 0.26, "tail": 2.20},
    "audio": {"enabled": True, "target_lufs": -14.0, "sfx": True, "seed": 7},
    "photo_crop": None,                  # [x0, y0, x1, y1] in source px; auto when null
    "lines": [],
    "slices": [],
}

FONT_CANDIDATES = [
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
    ("/System/Library/Fonts/STHeiti Light.ttc", 1),
    ("/System/Library/Fonts/Supplemental/NotoSansSC-Regular.otf", 0),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf", 0),
]


def deep_merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path):
    cfg = deep_merge(DEFAULTS, json.load(open(path, encoding="utf-8")))
    cfg["_config_dir"] = os.path.dirname(os.path.abspath(path))
    if not cfg["image"]:
        raise SystemExit("config.image is required")
    cfg["image"] = os.path.abspath(cfg["image"])
    cfg["out_dir"] = os.path.abspath(cfg["out_dir"] or os.path.join(cfg["_config_dir"], "out"))
    if not cfg["font"]["path"]:
        for fp, idx in FONT_CANDIDATES:
            if os.path.exists(fp):
                cfg["font"]["path"], cfg["font"]["index"] = fp, idx
                break
    if not cfg["font"]["path"]:
        raise SystemExit("no CJK font found; set font.path in the config")
    return cfg


def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode:
        sys.stderr.write((p.stderr or "")[-4000:] + "\n")
        raise SystemExit(f"command failed: {' '.join(cmd[:4])} ...")
    return (p.stdout or "") + (p.stderr or "")


def _flatten(im):
    """RGB, with any transparency composited onto white instead of black."""
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        return bg
    return im.convert("RGB")


def open_image(path):
    """Open any photo: honour EXIF rotation, and fall back to sips/ffmpeg for
    formats Pillow cannot read (HEIC, AVIF, some camera RAW)."""
    try:
        im = Image.open(path)
        im.load()
        return _flatten(ImageOps.exif_transpose(im))
    except Exception as first_error:
        tmp = tempfile.mkdtemp(prefix="photo-poem-src-")
        try:
            out = os.path.join(tmp, "source.jpg")
            for cmd in (["sips", "-s", "format", "jpeg", path, "--out", out],
                        ["ffmpeg", "-y", "-loglevel", "error", "-i", path, out]):
                if shutil.which(cmd[0]):
                    p = subprocess.run(cmd, capture_output=True)
                    if p.returncode == 0 and os.path.exists(out):
                        im = Image.open(out)
                        im.load()
                        return _flatten(ImageOps.exif_transpose(im))
            raise SystemExit(f"cannot read image {path}: {first_error}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# geometry + text layout
# --------------------------------------------------------------------------- #
class Poem:
    def __init__(self, cfg):
        self.cfg = cfg
        self.W = int(cfg["canvas"]["w"])
        self.H = int(cfg["canvas"]["h"])
        self.split = int(round(self.H * cfg["split"]))
        self.bg = tuple(cfg["bg"])
        self.ink = tuple(cfg["ink"])
        self.fs = self.W * cfg["font"]["size_ratio"]
        self.line_h = self.fs * cfg["font"]["line_height"]
        self.slice_h = self.fs * cfg["slice"]["height_ratio"]
        self.gap = float(cfg["slice"]["gap"])
        self.radius = int(cfg["slice"]["radius"])
        self.font = ImageFont.truetype(cfg["font"]["path"], round(self.fs * 4) / 4,
                                       index=int(cfg["font"]["index"]))
        self.src = open_image(cfg["image"])
        self.slices = [dict(s) for s in cfg["slices"]]
        if not self.slices:
            raise SystemExit("config.slices is empty")
        self.warnings = []
        self.box = self._photo_box()
        self.scale = self.W / (self.box[2] - self.box[0])
        self.photo = self.src.crop(self.box).resize(
            (self.W, self.H - self.split), Image.LANCZOS).convert("RGBA")
        self.lines = self._parse_lines(cfg["lines"])
        self.layout = self._layout()
        self._order = self._compute_order()
        self._timeline = self._compute_timeline()
        self._qa()

    # -- photo window ------------------------------------------------------ #
    def _photo_box(self):
        b = self.cfg.get("photo_crop")
        if b:
            return tuple(int(round(v)) for v in b)
        sw, sh = self.src.size
        target = self.W / (self.H - self.split)
        w = min(sw, int(round(sh * target)))
        h = min(sh, int(round(w / target)))
        x0, y0 = (sw - w) // 2, (sh - h) // 2
        return (x0, y0, x0 + w, y0 + h)

    # -- poem -------------------------------------------------------------- #
    def _parse_lines(self, raw):
        if not raw:
            raise SystemExit("config.lines is empty")
        used, out = set(), []
        for line in raw:
            items = []
            for chunk in re.split(r"(\{\d+\})", line):
                if not chunk:
                    continue
                m = re.fullmatch(r"\{(\d+)\}", chunk)
                if m:
                    idx = int(m.group(1))
                    if idx >= len(self.slices):
                        raise SystemExit(f"line references slice {{{idx}}} but only "
                                         f"{len(self.slices)} slices exist")
                    used.add(idx)
                    items += [("t", "（"), ("i", idx), ("t", "）")]
                else:
                    items.append(("t", chunk))
            out.append(items)
        missing = [i for i in range(len(self.slices)) if i not in used]
        if missing:
            self.warnings.append(f"slices never placed in the poem: {missing}")
        return out

    def _layout(self):
        f = self.font
        glyph = f.getbbox("国", anchor="ls")     # CJK ink box measured from the baseline
        glyph_cy = (glyph[1] + glyph[3]) / 2.0
        lines = []
        for items in self.lines:
            laid, total = [], 0.0
            for pos, (kind, val) in enumerate(items):
                if kind == "t":
                    w = f.getlength(val)
                    laid.append(dict(kind="t", text=val, w=w))
                else:
                    w, h = self.slice_display_size(val)
                    lead = self.gap if pos and items[pos - 1][0] == "t" else 0
                    trail = self.gap if pos + 1 < len(items) and items[pos + 1][0] == "t" else 0
                    laid.append(dict(kind="i", idx=val, w=w, h=h, lead=lead, trail=trail))
                    total += lead + trail
                total += w
            lines.append(dict(items=laid, w=total))
        ascent, descent = f.getmetrics()
        block_h = self.line_h * len(lines)
        top = self.split / 2 - block_h / 2
        for li, ln in enumerate(lines):
            ln["baseline"] = top + li * self.line_h + (self.line_h - (ascent + descent)) / 2 + ascent
            ln["x0"] = self.W / 2 - ln["w"] / 2
            x = ln["x0"]
            for it in ln["items"]:
                it["x"] = x + it.get("lead", 0)
                if it["kind"] == "i":
                    it["y"] = ln["baseline"] + glyph_cy - it["h"] / 2
                x += it["w"] + it.get("lead", 0) + it.get("trail", 0)
        self.text_layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(self.text_layer)
        for ln in lines:
            for it in ln["items"]:
                if it["kind"] == "t":
                    d.text((it["x"], ln["baseline"]), it["text"], font=f,
                           fill=self.ink + (255,), anchor="ls")
        return dict(lines=lines, block_h=block_h, top=top, glyph_cy=glyph_cy,
                    ascent=ascent, descent=descent)

    # -- slice geometry ---------------------------------------------------- #
    def slice_display_size(self, i):
        _, _, w, h = self.slices[i]["box"]
        return (self.slice_h * w / h, self.slice_h)

    def hole_rect(self, i):
        x, y, w, h = self.slices[i]["box"]
        return ((x - self.box[0]) * self.scale, self.split + (y - self.box[1]) * self.scale,
                w * self.scale, h * self.scale)

    def slot_rect(self, i):
        for ln in self.layout["lines"]:
            for it in ln["items"]:
                if it["kind"] == "i" and it["idx"] == i:
                    return (it["x"], it["y"], it["w"], it["h"])
        raise KeyError(i)

    def _compute_order(self):
        seen = []
        for ln in self.lines:
            for kind, val in ln:
                if kind == "i" and val not in seen:
                    seen.append(val)
        return seen

    def order(self):
        return self._order

    def patch(self, i, size):
        x, y, w, h = self.slices[i]["box"]
        im = self.src.crop((x, y, x + w, y + h)).resize(
            (max(1, int(round(size[0]))), max(1, int(round(size[1])))), Image.LANCZOS)
        mask = Image.new("L", im.size, 255)
        if self.radius > 0:
            ImageDraw.Draw(mask).rounded_rectangle(
                [0, 0, im.size[0] - 1, im.size[1] - 1], radius=self.radius, fill=255)
        out = im.convert("RGBA")
        out.putalpha(mask)
        return out

    # -- timeline ---------------------------------------------------------- #
    def _compute_timeline(self):
        tl = self.cfg["timeline"]
        events = {}
        for pos, i in enumerate(self.order()):
            t0 = tl["slice_start"] + pos * tl["slice_gap"]
            land = t0 + tl["pick"] + tl["fly"]
            events[i] = (t0, t0 + tl["pick"], land, land + tl["settle"])
        return events

    def timeline(self):
        return self._timeline

    def offset(self):
        tl = self.cfg["timeline"]
        return tl["slice_start"] - tl["cover_hold"] - tl["dissolve"]

    def end_time(self):
        return max(v[3] for v in self.timeline().values()) + self.cfg["timeline"]["tail"]

    def _qa(self):
        w = self.warnings
        tl = self.cfg["timeline"]
        step = tl["pick"] + tl["fly"] + tl["settle"]
        if tl["slice_gap"] < step - 1e-6:
            w.append(f"slice_gap {tl['slice_gap']:.2f}s is shorter than pick+fly+settle "
                     f"({step:.2f}s): slices will travel at the same time")
        if self.offset() < 0:
            w.append("cover_hold + dissolve is longer than slice_start: the no-cover cut "
                     "would start before the clip does")
        for li, ln in enumerate(self.layout["lines"]):
            ratio = ln["w"] / self.W
            if ratio > 0.95:
                w.append(f"line {li + 1} is {ratio * 100:.0f}% of the canvas wide — it may overflow")
            elif ratio < 0.40:
                w.append(f"line {li + 1} is only {ratio * 100:.0f}% wide — consider a longer line")
        holes = []
        for i in self.order():
            x, y, hw, hh = self.hole_rect(i)
            holes.append((i, x, y, hw, hh))
            if x < 12 or y < self.split + 12 or x + hw > self.W - 12 or y + hh > self.H - 12:
                w.append(f"slice {i} ({self.slices[i].get('key', i)}) touches the photo edge")
            dw, dh = self.slice_display_size(i)
            if not (0.04 <= dw / self.W <= 0.15):
                w.append(f"slice {i} display width {dw / self.W * 100:.1f}% is outside 4–15%")
            if not (1.3 <= dh / self.fs <= 2.1):
                w.append(f"slice {i} height {dh / self.fs:.2f}x text is outside 1.3–2.1x")
            up = self.slice_h / self.slices[i]["box"][3]
            if up > 1.3:
                w.append(f"slice {i} ({self.slices[i].get('key', i)}) is magnified {up:.1f}x — "
                         "pick a larger area, it will look soft")
        for a in range(len(holes)):
            for b in range(a + 1, len(holes)):
                _, x1, y1, w1, h1 = holes[a]
                _, x2, y2, w2, h2 = holes[b]
                if x1 < x2 + w2 and x2 < x1 + w1 and y1 < y2 + h2 and y2 < y1 + h1:
                    w.append(f"slices {holes[a][0]} and {holes[b][0]} overlap in the photo")
        for a in range(len(self.slices)):
            for b in range(a + 1, len(self.slices)):
                if tuple(self.slices[a]["box"]) == tuple(self.slices[b]["box"]):
                    w.append(f"slices {a} and {b} use the same box")


# --------------------------------------------------------------------------- #
# motion
# --------------------------------------------------------------------------- #
def ease_out(t):
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    t = min(1.0, max(0.0, t))
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def _bezier(p0, p1, p2, p3, k):
    u = 1 - k
    return (u ** 3 * p0[0] + 3 * u * u * k * p1[0] + 3 * u * k * k * p2[0] + k ** 3 * p3[0],
            u ** 3 * p0[1] + 3 * u * u * k * p1[1] + 3 * u * k * k * p2[1] + k ** 3 * p3[1])


def _tangent(p0, p1, p2, p3, k):
    u = 1 - k
    dx = 3 * u * u * (p1[0] - p0[0]) + 6 * u * k * (p2[0] - p1[0]) + 3 * k * k * (p3[0] - p2[0])
    dy = 3 * u * u * (p1[1] - p0[1]) + 6 * u * k * (p2[1] - p1[1]) + 3 * k * k * (p3[1] - p2[1])
    return math.atan2(dy, dx)


def flight_path(poem, i):
    hx, hy, hw, hh = poem.hole_rect(i)
    sx, sy, sw, sh = poem.slot_rect(i)
    p0 = (hx + hw / 2, hy + hh / 2)
    p3 = (sx + sw / 2, sy + sh / 2)
    dx, dy = p3[0] - p0[0], p3[1] - p0[1]
    p1 = (p0[0] + 0.10 * dx, p0[1] - 0.35 * abs(dy) - 45)
    p2 = (p3[0] - 0.22 * dx, p3[1] + 0.22 * abs(dy) + 25)
    return p0, p1, p2, p3


def _flight_pose(poem, i, u):
    hx, hy, hw, hh = poem.hole_rect(i)
    sx, sy, sw, sh = poem.slot_rect(i)
    p0, p1, p2, p3 = flight_path(poem, i)
    k = ease_in_out(u)
    cx, cy = _bezier(p0, p1, p2, p3, k)
    w = (hw * 1.06) + (sw - hw * 1.06) * k
    h = (hh * 1.06) + (sh - hh * 1.06) * k
    swell = 1.0 + 0.035 * math.sin(math.pi * u)
    ang = math.degrees(_tangent(p0, p1, p2, p3, k))
    return dict(cx=cx, cy=cy, w=w * swell, h=h * swell,
                bank=max(-6.5, min(6.5, 0.16 * ang)) * (1 - u ** 2),
                phi=ang, ks=0.05 * math.sin(math.pi * u), alpha=255)


def patch_poses(poem, i, t):
    tl = poem.cfg["timeline"]
    t0, pick_end, land, settle_end = poem.timeline()[i]
    hx, hy, hw, hh = poem.hole_rect(i)
    sx, sy, sw, sh = poem.slot_rect(i)
    if t < t0:
        return None
    if t < pick_end:                               # plucked off the page
        s = 1.0 + 0.06 * ease_out((t - t0) / tl["pick"])
        return [dict(cx=hx + hw / 2, cy=hy + hh / 2, w=hw * s, h=hh * s,
                     bank=0.0, phi=0.0, ks=0.0, alpha=255)]
    if t < land:                                   # travelling
        u = (t - pick_end) / tl["fly"]
        poses = [_flight_pose(poem, i, u)]
        if 0.08 < u < 0.92:
            for du, a in ((0.035, 78), (0.07, 38)):
                if u - du > 0:
                    g = _flight_pose(poem, i, u - du)
                    g["alpha"] = a
                    poses.append(g)
        return poses
    if t < settle_end:                             # springy settle
        u = min(1.0, (t - land) / tl["settle"])
        dip = (1 - ease_out(u)) ** 2
        return [dict(cx=sx + sw / 2, cy=sy + sh / 2,
                     w=sw * (1 + 0.035 * dip), h=sh * (1 + 0.035 * dip),
                     bank=1.6 * dip * math.sin(7.5 * u), phi=0.0, ks=0.0, alpha=255)]
    return [dict(cx=sx + sw / 2, cy=sy + sh / 2, w=sw, h=sh,
                 bank=0.0, phi=0.0, ks=0.0, alpha=255)]


def _affine(p, bank_deg, ks, phi_deg):
    """Stretch along the travel direction, then bank the slice."""
    if abs(ks) < 0.004 and abs(bank_deg) < 0.06:
        return p
    w, h = p.size
    cb, sb = math.cos(math.radians(bank_deg)), math.sin(math.radians(bank_deg))
    cp, sp = math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg))
    rot_b, rot_p, rot_n = ((cb, -sb), (sb, cb)), ((cp, -sp), (sp, cp)), ((cp, sp), (-sp, cp))
    scl = ((1 + ks, 0.0), (0.0, 1 - ks))

    def mul(a, b):
        return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2))
                     for i in range(2))

    m = mul(rot_b, mul(rot_p, mul(scl, rot_n)))
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    inv = ((m[1][1] / det, -m[0][1] / det), (-m[1][0] / det, m[0][0] / det))
    cx, cy = w / 2, h / 2
    corners = ((-cx, -cy), (cx, -cy), (cx, cy), (-cx, cy))
    out = [(m[0][0] * x + m[0][1] * y, m[1][0] * x + m[1][1] * y) for x, y in corners]
    ow = int(math.ceil(max(o[0] for o in out) - min(o[0] for o in out))) + 2
    oh = int(math.ceil(max(o[1] for o in out) - min(o[1] for o in out))) + 2
    tx = cx - (inv[0][0] * ow / 2 + inv[0][1] * oh / 2)
    ty = cy - (inv[1][0] * ow / 2 + inv[1][1] * oh / 2)
    return p.transform((ow, oh), Image.AFFINE,
                       (inv[0][0], inv[0][1], tx, inv[1][0], inv[1][1], ty),
                       resample=Image.BICUBIC)


# --------------------------------------------------------------------------- #
# frame compositing
# --------------------------------------------------------------------------- #
def compose(poem, t):
    root = Image.new("RGBA", (poem.W, poem.H), poem.bg + (255,))
    root.alpha_composite(poem.photo, (0, poem.split))

    tl, mk = poem.cfg["timeline"], poem.cfg["marker"]
    marks = Image.new("RGBA", (poem.W, poem.H), (0, 0, 0, 0))
    d = ImageDraw.Draw(marks)
    for i in poem.order():
        t0 = poem.timeline()[i][0]
        k = (t - t0) / tl["wipe"]
        if k <= 0:
            continue
        a = ease_out(min(1.0, k))
        s = 0.55 + 0.45 * a
        x, y, w, h = poem.hole_rect(i)
        box = [x + (w - w * s) / 2, y + (h - h * s) / 2,
               x + (w + w * s) / 2 - 1, y + (h + h * s) / 2 - 1]
        d.rectangle(box, fill=poem.bg + (int(mk["fill_alpha"] * a),))
        d.rectangle(box, outline=poem.bg + (int(mk["edge_alpha"] * a),), width=mk["edge_width"])
    root.alpha_composite(marks)

    root.alpha_composite(poem.text_layer)

    for i in poem.order():
        poses = patch_poses(poem, i, t)
        if not poses:
            continue
        for pose in reversed(poses):
            p = _affine(poem.patch(i, (pose["w"], pose["h"])),
                        pose["bank"], pose["ks"], pose["phi"])
            if pose["alpha"] < 255:
                p.putalpha(p.getchannel("A").point(lambda v: v * pose["alpha"] // 255))
            root.alpha_composite(p, (int(round(pose["cx"] - p.width / 2)),
                                     int(round(pose["cy"] - p.height / 2))))
    return root.convert("RGB")


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def open_encoder(poem, out, wav=None, gain_db=None, trim=0.0):
    """Start an ffmpeg that consumes raw RGB frames on stdin.

    `wav` is an optional score, `trim` the seconds to skip on it (the no-cover cut).
    Returns the Popen handle; the caller writes frames and closes stdin.
    """
    cfg = poem.cfg
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{poem.W}x{poem.H}",
           "-framerate", str(cfg["fps"]), "-i", "-"]
    if wav:
        if trim:
            cmd += ["-ss", f"{trim:.3f}"]
        cmd += ["-i", wav, "-map", "0:v", "-map", "1:a"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p"]
    if wav:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-af", f"volume={gain_db:.2f}dB", "-shortest"]
    cmd += ["-movflags", "+faststart", out]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def write_frame(proc, frame_bytes):
    try:
        proc.stdin.write(frame_bytes)
    except (BrokenPipeError, ValueError):
        raise SystemExit("encoder stopped early — see the ffmpeg error above")


def measure_lufs(path):
    out = run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
               "-af", "ebur128=framelog=quiet", "-f", "null", "-"])
    m = re.search(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", out)
    return float(m.group(1)) if m else -14.0


def write_wav(path, mix):
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(_score().SR)
        w.writeframes((mix * 32767).astype("<i2").tobytes())


def score_signals(poem, duration, cfg, sfx=None, stems=False):
    """Build the score from the same timeline the picture uses."""
    score = _score()
    offset, timed = poem.offset(), poem.timeline()
    picks = [timed[i][0] - offset for i in poem.order()]
    lands = [timed[i][2] - offset for i in poem.order()]
    flights = []
    for i in poem.order():
        p = flight_path(poem, i)
        flights.append(dict(
            t0=timed[i][1] - offset, t1=timed[i][2] - offset,
            pan=(lambda u, p=p: max(-0.75, min(0.75,
                (_bezier(*p, ease_in_out(u))[0] / poem.W * 2 - 1) * 0.8)))))
    return score.build_mix(duration, picks, lands, flights, seed=cfg["audio"]["seed"],
                           melody=score.melody_for(lands),
                           sfx=cfg["audio"]["sfx"] if sfx is None else sfx,
                           return_stems=stems)


def build(cfg_path, dump_frames=None):
    cfg = load_config(cfg_path)
    poem = Poem(cfg)
    name, fps = cfg["basename"], cfg["fps"]
    os.makedirs(cfg["out_dir"], exist_ok=True)
    tmp = os.path.join(cfg["out_dir"], f".work-{name}")
    os.makedirs(tmp, exist_ok=True)
    h = cfg["canvas"]["h"]
    out_cover = os.path.join(cfg["out_dir"], f"{name}-1080x{h}.mp4")
    out_plain = os.path.join(cfg["out_dir"], f"{name}-nocover-1080x{h}.mp4")
    tl = cfg["timeline"]
    n_hold = int(round(tl["cover_hold"] * fps))
    n_diss = int(round(tl["dissolve"] * fps))
    start = poem.offset() + tl["cover_hold"] + tl["dissolve"]
    n_src = int(round((poem.end_time() - start) * fps))
    cover_dur = (n_src + n_hold + n_diss) / fps

    wav, gain = None, None
    if cfg["audio"]["enabled"]:
        wav = os.path.join(tmp, "mix.wav")
        write_wav(wav, score_signals(poem, cover_dur, cfg))
        gain = cfg["audio"]["target_lufs"] - measure_lufs(wav)

    finish = compose(poem, poem.end_time() - 0.05)
    first = compose(poem, start)
    cover_enc = open_encoder(poem, out_cover, wav, gain)
    plain_enc = open_encoder(poem, out_plain, wav, gain,
                             trim=max(0.0, poem.offset()))
    if dump_frames:
        os.makedirs(dump_frames, exist_ok=True)
    try:
        # the cover version opens on the finished frame, then dissolves into the blanks
        fb = finish.tobytes()
        for _ in range(n_hold):
            write_frame(cover_enc, fb)
        for f in range(n_diss):
            write_frame(cover_enc, Image.blend(finish, first, ease_in_out((f + 1) / n_diss)).tobytes())
        for f in range(n_src):
            frame = compose(poem, start + f / fps)
            data = frame.tobytes()
            write_frame(cover_enc, data)
            write_frame(plain_enc, data)
            if dump_frames:
                frame.save(os.path.join(dump_frames, f"{f:05d}.png"), compress_level=1)
    finally:
        for enc in (cover_enc, plain_enc):
            if enc.stdin and not enc.stdin.closed:
                enc.stdin.close()
        codes = [enc.wait() for enc in (cover_enc, plain_enc)]
        shutil.rmtree(tmp, ignore_errors=True)

    finish.save(os.path.join(cfg["out_dir"], f"{name}-cover-1080x{h}.jpg"), quality=94)
    report(poem, out_cover, out_plain, cover_dur)
    if any(codes):
        raise SystemExit(f"ffmpeg exited with {codes}")
    return poem


def report(poem, out_cover, out_plain, dur):
    print("=" * 70)
    print("poem: " + "".join(v for ln in poem.lines for k, v in ln if k == "t"))
    for li, ln in enumerate(poem.layout["lines"]):
        print(f"  line {li + 1}: {ln['w'] / poem.W * 100:4.1f}% of canvas wide")
    print(f"photo window {poem.box}  (scale {poem.scale:.3f})")
    for i in poem.order():
        dw, dh = poem.slice_display_size(i)
        x, y, w, h = poem.hole_rect(i)
        print(f"  slice {i} {str(poem.slices[i].get('key', i)):<10s} in-text {dw:5.0f}x{dh:.0f} "
              f"({dw / poem.W * 100:4.1f}% wide)  marker ({x:.0f},{y:.0f}) {w:.0f}x{h:.0f}")
    print(f"duration {dur:.2f}s -> {os.path.basename(out_cover)} | {os.path.basename(out_plain)}")
    if poem.warnings:
        print("-" * 70)
        for msg in poem.warnings:
            print("  ! " + msg)
    else:
        print("checks: ok")
    print("=" * 70)


def stills(cfg_path, times, out=None):
    poem = Poem(load_config(cfg_path))
    out = out or os.path.join(os.path.dirname(os.path.abspath(cfg_path)), "stills")
    os.makedirs(out, exist_ok=True)
    for t in times:
        compose(poem, t).save(os.path.join(out, f"t{t:07.2f}.png"))
    print("wrote stills to", out)


def main():
    ap = argparse.ArgumentParser(description="photo -> poem-with-blanks video")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="render the video")
    b.add_argument("config")
    b.add_argument("--dump-frames", metavar="DIR",
                   help="also write the source-timeline frames as PNGs (debugging)")
    s = sub.add_parser("stills", help="render single frames for a visual check")
    s.add_argument("config")
    s.add_argument("--times", default="1.2,2.6,9.6")
    s.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.config, a.dump_frames)
    else:
        stills(a.config, [float(x) for x in a.times.split(",")], a.out)


if __name__ == "__main__":
    main()
