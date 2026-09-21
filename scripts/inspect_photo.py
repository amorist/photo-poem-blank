#!/usr/bin/env python3
"""Read a photo's geometry and mark crop candidates, so slice boxes can be chosen by eye.

    python3 inspect_photo.py photo.jpg --grid
    python3 inspect_photo.py photo.jpg --boxes "blossom:690,302,92,60;sign:100,62,150,70"

Writes an annotated overlay (photo window + 100px graph paper) and a contact sheet
of the candidate slices rendered at the size they will appear inside the poem.
"""
import argparse
import os
import re

from PIL import Image, ImageDraw, ImageFont

from poem_video import DEFAULTS                        # noqa: E402  (same folder)


def parse_boxes(spec):
    out = []
    for chunk in filter(None, (c.strip() for c in spec.split(";"))):
        name, nums = chunk.split(":", 1)
        x, y, w, h = (int(round(float(v))) for v in re.split(r"[,\s]+", nums.strip()))
        out.append((name.strip(), (x, y, w, h)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--canvas", default="1080x1440")
    ap.add_argument("--split", type=float, default=0.48)
    ap.add_argument("--crop", help="photo window x0,y0,x1,y1 (default: auto)")
    ap.add_argument("--boxes", default="", help='name:x,y,w,h separated by ";"')
    ap.add_argument("--grid", action="store_true",
                    help="draw a labelled 100px coordinate grid over the photo")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    cw, ch = (int(v) for v in a.canvas.lower().split("x"))
    cfg = {**DEFAULTS, "image": a.image, "canvas": {"w": cw, "h": ch}, "split": a.split,
           "slices": [{"key": "probe", "box": [0, 0, 10, 10]}]}
    if a.crop:
        cfg["photo_crop"] = [int(float(v)) for v in a.crop.split(",")]
    src = Image.open(a.image).convert("RGB")
    sw, sh = src.size
    target = cw / (ch * (1 - a.split))
    if cfg.get("photo_crop"):
        x0, y0, x1, y1 = cfg["photo_crop"]
    else:
        w = min(sw, int(round(sh * target)))
        h = min(sh, int(round(w / target)))
        x0, y0 = (sw - w) // 2, (sh - h) // 2
        x1, y1 = x0 + w, y0 + h
    scale = cw / (x1 - x0)
    outdir = a.outdir or os.path.join(os.getcwd(), "poem-inspect")
    os.makedirs(outdir, exist_ok=True)

    print(f"source {sw}x{sh}  aspect {sw / sh:.3f}")
    print(f"canvas {cw}x{ch}  split {a.split:.2f}  ->  photo window needs aspect {target:.3f}")
    print(f"photo window (photo_crop): [{x0}, {y0}, {x1}, {y1}]   {x1 - x0}x{y1 - y0}"
          f"   -> displayed at scale {scale:.3f}")
    print(f"usable slice width inside the poem: {0.04 * cw:.0f}-{0.14 * cw:.0f}px"
          f"  (height {0.032 * cw * 1.6:.0f}px at the default 1.6x text)")

    ov = src.copy()
    d = ImageDraw.Draw(ov)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", max(11, sw // 90))
    except OSError:
        f = ImageFont.load_default()
    if a.grid:
        step = 100
        for gx in range(0, sw, step):
            d.line([(gx, 0), (gx, sh)], fill=(255, 0, 0), width=1)
            d.text((gx + 3, 3), str(gx), fill=(255, 40, 40), font=f)
        for gy in range(0, sh, step):
            d.line([(0, gy), (sw, gy)], fill=(255, 0, 0), width=1)
            d.text((3, gy + 2), str(gy), fill=(255, 40, 40), font=f)
    d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(0, 160, 255), width=3)
    for name, (bx, by, bw, bh) in parse_boxes(a.boxes):
        d.rectangle([bx, by, bx + bw - 1, by + bh - 1], outline=(0, 255, 180), width=3)
        d.text((bx + 4, by + 4), name, fill=(255, 255, 0), font=f)
    grid_path = os.path.join(outdir, "grid.png")
    ov.save(grid_path)
    print("overlay ->", grid_path)

    boxes = parse_boxes(a.boxes)
    if boxes:
        dh = 0.032 * cw * 1.6
        tiles = []
        for name, (bx, by, bw, bh) in boxes:
            dw = dh * bw / bh
            tile = src.crop((bx, by, bx + bw, by + bh)).resize(
                (int(round(dw)), int(round(dh))), Image.LANCZOS)
            tiles.append((name, tile, dw, dh))
        pad, label = 16, 22
        sheet = Image.new("RGB", (sum(int(t[2]) * 3 + pad * 2 for t in tiles), int(dh * 3) + label * 2),
                          (250, 250, 250))
        sd = ImageDraw.Draw(sheet)
        x = pad
        for name, tile, dw, dh2 in tiles:
            big = tile.resize((int(dw * 3), int(dh * 3)), Image.NEAREST)
            sheet.paste(big, (x, label))
            sd.text((x, 4), f"{name} {dw:.0f}x{dh:.0f}", fill=(0, 0, 0), font=f)
            x += int(dw * 3) + pad * 2
        sheet_path = os.path.join(outdir, "boxes.png")
        sheet.save(sheet_path)
        print("candidate slices at real in-text size (3x pixels) ->", sheet_path)


if __name__ == "__main__":
    main()
