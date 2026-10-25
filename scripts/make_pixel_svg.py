"""
Build a COLOR pixel-art portrait (terminal-window SVG) from the source photo.

Pipeline (Pillow + numpy only):
  1. grayscale-based background flood-fill + connected-component cleanup to
     isolate the head (same approach as prep_photo.py), applied to the COLOR image
  2. enhance colour/contrast/sharpness so eyes, brows and beard read at low res
  3. downsample to a COLS x ROWS grid of coloured cells
  4. emit an SVG: dark terminal window with the portrait as coloured squares that
     pop in row by row (SMIL/CSS, which GitHub runs inside <img>), then hold

    python scripts/make_pixel_svg.py [input] [output.svg] [--png preview.png]

Env: COLS, SAT, CONTRAST, SHARP, CROP_SCALE, TOP_BIAS, BG_THRESH, MIN_BLOB.
"""
import os
import sys
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
INP = args[0] if len(args) > 0 else os.path.join(HERE, "..", "source-photo.jpg")
OUT = args[1] if len(args) > 1 else os.path.join(HERE, "..", "kunal-portrait.svg")
PNG = None
if "--png" in sys.argv:
    PNG = sys.argv[sys.argv.index("--png") + 1]

COLS = int(os.environ.get("COLS", 82))
SAT = float(os.environ.get("SAT", 1.22))
CONTRAST = float(os.environ.get("CONTRAST", 1.28))
SHARP = float(os.environ.get("SHARP", 1.7))
CROP_SCALE = float(os.environ.get("CROP_SCALE", 0.84))
TOP_BIAS = float(os.environ.get("TOP_BIAS", 0.18))
BG_THRESH = int(os.environ.get("BG_THRESH", 34))
MIN_BLOB = int(os.environ.get("MIN_BLOB", 700))

# ---- 1. load + crop ----
src = Image.open(INP).convert("RGB")
w, h = src.size
side = int(min(w, h) * CROP_SCALE)
left = (w - side) // 2
top = max(0, min(int((h - side) * TOP_BIAS), h - side))
src = src.crop((left, top, left + side, top + side))
W, H = src.size
gray = src.convert("L")

# ---- 2. background flood-fill -> mask ----
flood = gray.copy()
for s in [(2, 2), (W - 3, 2), (2, H - 3), (W - 3, H - 3),
          (W // 2, 2), (2, H // 2), (W - 3, H // 2)]:
    ImageDraw.floodfill(flood, s, 255, thresh=BG_THRESH)
garr = np.asarray(gray, np.float32)
bg = np.asarray(flood, np.float32) >= 254.5
band = max(6, W // 12)
border = np.zeros((H, W), bool)
border[:, :band] = border[:, -band:] = border[:band, :] = True
bg |= (garr > 200) & border

# keep only the largest subject blob (drop specks)
subj = ~bg
labels = np.zeros((H, W), np.int32)
sizes = [0]
cur = 0
for y0 in range(H):
    for x0 in range(W):
        if subj[y0, x0] and labels[y0, x0] == 0:
            cur += 1
            q = deque([(y0, x0)])
            labels[y0, x0] = cur
            n = 0
            while q:
                y, x = q.popleft()
                n += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and subj[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = cur
                        q.append((ny, nx))
            sizes.append(n)
keep = np.array([s >= MIN_BLOB for s in sizes])
subj = subj & keep[labels] if cur else subj
if os.environ.get("NOBG", "0") != "0":          # full frame: keep every cell
    subj = np.ones((H, W), bool)

# ---- 3. enhance + downsample to grid ----
img = ImageEnhance.Color(src).enhance(SAT)
img = ImageEnhance.Contrast(img).enhance(CONTRAST)
img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(SHARP * 100), threshold=1))

ROWS = round(COLS * H / W)
small = np.asarray(img.resize((COLS, ROWS), Image.LANCZOS), np.float32)
# downsample the subject mask: a cell is kept if most of it is subject
mask_small = np.asarray(
    Image.fromarray((subj * 255).astype(np.uint8)).resize((COLS, ROWS), Image.BILINEAR)
) > 110

# ---- 4. emit SVG ----
BG = "#0d1117"; BG2 = "#111722"; FRAME = "#30363d"; TITLE = "#7d8590"; INK = "#c9d1d9"
PAD = 20; TITLEBAR_H = 30; STATUS_H = 30
ART_W = 800
CELL = ART_W / COLS
ART_H = CELL * ROWS
CANVAS_W = ART_W + PAD * 2
CANVAS_H = TITLEBAR_H + ART_H + STATUS_H + PAD
art_top = TITLEBAR_H + PAD * 0.35

REVEAL = 3.2
rows_delay = REVEAL / ROWS

p = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W:.0f}" height="{CANVAS_H:.0f}" '
    f'viewBox="0 0 {CANVAS_W:.0f} {CANVAS_H:.0f}" '
    f'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">',
    '<style>'
    '.rw{opacity:0;animation:pop .45s ease-out both}'
    '@keyframes pop{0%{opacity:0;transform:translateY(-6px)}100%{opacity:1;transform:translateY(0)}}'
    '@media (prefers-reduced-motion:reduce){.rw{opacity:1!important;transform:none!important;animation:none!important}}'
    '</style>',
    f'<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">'
    f'<stop offset="0" stop-color="{BG2}"/><stop offset="1" stop-color="{BG}"/></linearGradient></defs>',
    f'<rect width="{CANVAS_W:.0f}" height="{CANVAS_H:.0f}" rx="12" fill="url(#bg)"/>',
    f'<rect x="0.5" y="0.5" width="{CANVAS_W-1:.0f}" height="{CANVAS_H-1:.0f}" rx="12" fill="none" stroke="{FRAME}"/>',
    f'<line x1="0" y1="{TITLEBAR_H}" x2="{CANVAS_W:.0f}" y2="{TITLEBAR_H}" stroke="{FRAME}"/>',
]
for i, dot in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
    p.append(f'<circle cx="{PAD + i*16}" cy="{TITLEBAR_H/2}" r="5" fill="{dot}"/>')
p.append(f'<text x="{CANVAS_W/2:.0f}" y="{TITLEBAR_H/2 + 4:.0f}" fill="{TITLE}" font-size="12" '
         f'text-anchor="middle">kunal@github: ~$ ./portrait.sh</text>')

gap = 0.5  # hairline gap between pixels for a crisp grid look
for ry in range(ROWS):
    cells = []
    for cx in range(COLS):
        if not mask_small[ry, cx]:
            continue
        r, g, b = small[ry, cx]
        x = PAD + cx * CELL
        y = art_top + ry * CELL
        cells.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{CELL-gap:.1f}" height="{CELL-gap:.1f}" '
                     f'rx="1" fill="#{int(r):02x}{int(g):02x}{int(b):02x}"/>')
    if cells:
        p.append(f'<g class="rw" style="animation-delay:{ry*rows_delay:.2f}s">{"".join(cells)}</g>')

# status bar
sly = TITLEBAR_H + ART_H + PAD * 0.35
sy = sly + 19
p.append(f'<line x1="0" y1="{sly:.1f}" x2="{CANVAS_W:.0f}" y2="{sly:.1f}" stroke="{FRAME}"/>')
p.append(f'<text x="{PAD}" y="{sy:.1f}" fill="{TITLE}" font-size="13">'
         f'kunal@github:~$ whoami <tspan fill="{INK}">Kunal Kumar</tspan></text>')
p.append("</svg>")
svg = "".join(p)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(svg)
print("wrote", OUT, len(svg) // 1024, "KB;", f"{COLS}x{ROWS} grid")

# ---- optional PNG preview (faithful raster of the same grid) ----
if PNG:
    scale = 10
    pw, ph = int(CANVAS_W * scale / 2), int(CANVAS_H * scale / 2)
    prev = Image.new("RGB", (pw, ph), (13, 17, 23))
    d = ImageDraw.Draw(prev)
    sc = scale / 2
    for ry in range(ROWS):
        for cx in range(COLS):
            if not mask_small[ry, cx]:
                continue
            r, g, b = small[ry, cx]
            x = (PAD + cx * CELL) * sc
            y = (art_top + ry * CELL) * sc
            d.rectangle([x, y, x + (CELL - gap) * sc, y + (CELL - gap) * sc],
                        fill=(int(r), int(g), int(b)))
    prev.save(PNG)
    print("wrote preview", PNG, prev.size)
