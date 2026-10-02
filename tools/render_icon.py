#!/usr/bin/env python3
"""Render the kestrel-audio icon set from the Kestrel falcon tile.

The audio member of the Kestrel family: the same falcon tile as the camera app (cut to a
crisp rounded square, transparent corners) with a waveform of rounded bars tucked into the
empty lower-right of the plate, clear of the falcon's face. The falcon is never redrawn;
the bars are drawn at 4x and averaged down.

Outputs (all inside assets/, deterministic: no timestamps, fixed PNG options):
    icon-512.png  favicon-32.png  favicon.ico (16/32/48)  icon.svg

    python3 tools/render_icon.py        # needs Pillow + numpy, see requirements-icon.txt
"""
from __future__ import annotations

import base64
import io
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
SOURCE = ASSETS / "kestrel-icon-source.png"

SIZE = 512
PAD = 0.04          # transparent margin, same as the Kestrel brand icon
CORNER = 0.215      # corner radius as a share of tile width (matches the artwork)
SS = 4              # supersampling for drawn parts

# Waveform geometry, in fractions of the tile (0..1 of its width/height).
# Bars are centred on cy; heights are shares of `half` (peak = 1.0, centred).
# The fat variant (fewer, bolder bars) is used for 16/32 px so the motif survives.
WAVE = dict(heights=(0.36, 0.66, 0.88, 1.00, 0.88, 0.66, 0.36),
            x0=0.535, x1=0.925, cy=0.858, half=0.110, fill=0.70)
WAVE_SMALL = dict(heights=(0.55, 0.85, 1.00, 0.85, 0.55),
                  x0=0.50, x1=0.935, cy=0.835, half=0.135, fill=0.80)
WAVE_TINY = dict(heights=(0.70, 1.00, 0.70),
                 x0=0.50, x1=0.94, cy=0.83, half=0.150, fill=0.72)
SMALL_MAX = 32  # icon sizes up to this use WAVE_SMALL, up to TINY_MAX use WAVE_TINY
TINY_MAX = 16

# Rose Pine (moon-ish dark) accents: foam body, rose peak, with a light top for a soft gloss.
FOAM = (156, 207, 216)     # #9CCFD8
FOAM_LIGHT = (200, 232, 238)
ROSE = (235, 111, 146)     # #EB6F92
ROSE_LIGHT = (246, 168, 190)
GOLD = (246, 193, 119)     # #F6C177


def tile() -> Image.Image:
    """Solid tile cropped to its edges with a clean rounded-square alpha (as render_brand.py)."""
    src = Image.open(SOURCE).convert("RGBA")
    solid = src.getchannel("A").point(lambda a: 255 if a > 235 else 0)
    left, top, right, bottom = solid.getbbox()
    side = max(right - left, bottom - top)
    cx, cy = (left + right) // 2, (top + bottom) // 2
    x0, y0 = cx - side // 2, cy - side // 2
    crop = src.crop((x0, y0, x0 + side, y0 + side))
    mask = Image.new("L", (side * SS, side * SS), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, side * SS - 1, side * SS - 1), radius=int(side * SS * CORNER), fill=255)
    mask = mask.resize((side, side), Image.LANCZOS)
    crop.putalpha(ImageChops.multiply(crop.getchannel("A"), mask))
    return crop


def bar_specs(inner: int, wave: dict = WAVE) -> list[dict]:
    """Bar rectangles in pixels of an inner x inner tile, plus per-bar colours."""
    heights = wave["heights"]
    n = len(heights)
    pitch = (wave["x1"] - wave["x0"]) / n
    width = pitch * wave["fill"]
    mid = n // 2
    out = []
    for i, h in enumerate(heights):
        cx = wave["x0"] + pitch * (i + 0.5)
        half = wave["half"] * h
        peak = i == mid
        out.append(dict(
            x=(cx - width / 2) * inner, y=(wave["cy"] - half) * inner,
            w=width * inner, h=2 * half * inner, r=width * inner / 2,
            top=ROSE_LIGHT if peak else FOAM_LIGHT, bottom=ROSE if peak else FOAM))
    return out


def draw_bars(inner: int, wave: dict = WAVE) -> Image.Image:
    """RGBA layer (inner x inner) with the vertical-gradient rounded bars, 4x supersampled."""
    big = inner * SS
    layer = np.zeros((big, big, 4), np.float32)
    for b in bar_specs(inner, wave):
        m = Image.new("L", (big, big), 0)
        ImageDraw.Draw(m).rounded_rectangle(
            (b["x"] * SS, b["y"] * SS, (b["x"] + b["w"]) * SS - 1, (b["y"] + b["h"]) * SS - 1),
            radius=b["r"] * SS, fill=255)
        a = np.asarray(m, np.float32) / 255.0
        ys = np.arange(big, dtype=np.float32)
        t = np.clip((ys - b["y"] * SS) / (b["h"] * SS), 0, 1)[:, None]
        col = (np.array(b["top"], np.float32)[None, None, :] * (1 - t[..., None])
               + np.array(b["bottom"], np.float32)[None, None, :] * t[..., None])
        col = np.broadcast_to(col, (big, big, 3))
        layer[..., :3] = np.where(a[..., None] > 0, col, layer[..., :3])
        layer[..., 3] = np.maximum(layer[..., 3], a)
    rgb = np.rint(layer[..., :3]).clip(0, 255).astype(np.uint8)
    alpha = np.rint(layer[..., 3] * 255.0).clip(0, 255).astype(np.uint8)
    # premultiplied downscale so edge colour does not bleed black
    pm = np.dstack([rgb.astype(np.float32) * alpha[..., None] / 255.0, alpha.astype(np.float32)])
    pm = pm.reshape(inner, SS, inner, SS, 4).mean(axis=(1, 3))
    a = pm[..., 3]
    col = np.where(a[..., None] > 0, pm[..., :3] / np.maximum(a[..., None], 1e-6) * 255.0, 0.0)
    out = np.dstack([np.rint(col).clip(0, 255), np.rint(a)]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def shadow(bars: Image.Image) -> Image.Image:
    """A soft dark halo under the bars so they separate from the lavender wing edge."""
    from PIL import ImageFilter
    a = bars.getchannel("A").filter(ImageFilter.GaussianBlur(bars.width * 0.012))
    a = a.point(lambda v: int(v * 0.55))
    out = Image.new("RGBA", bars.size, (18, 14, 28, 0))
    out.putalpha(a)
    return out


def compose_tile(inner: int, with_bars: bool = True, wave: dict = WAVE) -> Image.Image:
    base = tile().resize((inner, inner), Image.LANCZOS)
    if with_bars:
        bars = draw_bars(inner, wave)
        base.alpha_composite(shadow(bars))
        base.alpha_composite(bars)
    return base


def icon(size: int, with_bars: bool = True) -> Image.Image:
    pad = round(size * PAD)
    inner = size - 2 * pad
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    wave = WAVE_TINY if size <= TINY_MAX else WAVE_SMALL if size <= SMALL_MAX else WAVE
    canvas.alpha_composite(compose_tile(inner, with_bars, wave), (pad, pad))
    return canvas


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True, compress_level=9)
    return buf.getvalue()


def ico_bytes(images: list[Image.Image]) -> bytes:
    """Hand-built ICO of PNG entries (byte-stable, each size rendered from the master)."""
    blobs = [png_bytes(im) for im in images]
    head = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = b""
    for im, blob in zip(images, blobs):
        w = im.width % 256  # 256 is stored as 0
        entries += struct.pack("<BBBBHHII", w, w, 0, 0, 1, 32, len(blob), offset)
        offset += len(blob)
    return head + entries + b"".join(blobs)


def svg_text() -> str:
    pad = round(SIZE * PAD)
    inner = SIZE - 2 * pad
    tile_png = base64.b64encode(png_bytes(compose_tile(inner, with_bars=False))).decode()
    def hx(c): return "#%02X%02X%02X" % c
    defs, rects = [], []
    for i, b in enumerate(bar_specs(inner)):
        gid = f"g{i}"
        defs.append(f'<linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
                    f'<stop offset="0" stop-color="{hx(b["top"])}"/>'
                    f'<stop offset="1" stop-color="{hx(b["bottom"])}"/></linearGradient>')
        rects.append(f'<rect x="{b["x"]:.2f}" y="{b["y"]:.2f}" width="{b["w"]:.2f}" '
                     f'height="{b["h"]:.2f}" rx="{b["r"]:.2f}" fill="url(#{gid})"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {SIZE} {SIZE}" width="{SIZE}" height="{SIZE}">\n'
        f'<title>Kestrel Audio</title>\n<defs>{"".join(defs)}'
        f'<filter id="halo" x="-10%" y="-10%" width="120%" height="120%">'
        f'<feGaussianBlur stdDeviation="{inner * 0.012:.2f}"/></filter></defs>\n'
        f'<image x="{pad}" y="{pad}" width="{inner}" height="{inner}" '
        f'xlink:href="data:image/png;base64,{tile_png}"/>\n'
        f'<g transform="translate({pad} {pad})">\n'
        f'<g filter="url(#halo)" fill="#120E1C" opacity="0.55">'
        + "".join(f'<rect x="{b["x"]:.2f}" y="{b["y"]:.2f}" width="{b["w"]:.2f}" '
                  f'height="{b["h"]:.2f}" rx="{b["r"]:.2f}"/>' for b in bar_specs(inner))
        + '</g>\n' + "\n".join(rects) + '\n</g>\n</svg>\n')


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    master = icon(SIZE)
    (ASSETS / "icon-512.png").write_bytes(png_bytes(master))
    (ASSETS / "favicon-32.png").write_bytes(png_bytes(icon(32)))
    (ASSETS / "favicon.ico").write_bytes(ico_bytes([icon(s) for s in (16, 32, 48)]))
    (ASSETS / "icon.svg").write_text(svg_text(), encoding="utf-8", newline="\n")
    print(f"wrote icon set to {ASSETS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
