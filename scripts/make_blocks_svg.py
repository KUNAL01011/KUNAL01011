"""
Monochrome pixel-block portrait (terminal-window SVG) with drawn-on glasses.

Pillow + numpy only. Same isolation approach as the colour version, but renders
the head as grey blocks (light ink on dark), sharpens the eyes, and overlays a
pixel-drawn eyeglasses frame positioned by face proportions (tunable via env).

    python scripts/make_blocks_svg.py [input] [output.svg] [--png preview.png]

Env: COLS, CONTRAST, SHARP, LEVELS, CROP_SCALE, TOP_BIAS, BG_THRESH, MIN_BLOB,
     EYE_ROW, EYE_HALF, LENS_RX, LENS_RY, GLASSES(1/0), EYE_SHARP(1/0).
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
PNG = sys.argv[sys.argv.index("--png") + 1] if "--png" in sys.argv else None

COLS = int(os.environ.get("COLS", 120))
CONTRAST = float(os.environ.get("CONTRAST", 1.42))
SHARP = float(os.environ.get("SHARP", 1.9))
LEVELS = int(os.environ.get("LEVELS", 9))          # quantise greys -> block feel
CROP_SCALE = float(os.environ.get("CROP_SCALE", 0.84))
TOP_BIAS = float(os.environ.get("TOP_BIAS", 0.18))
BG_THRESH = int(os.environ.get("BG_THRESH", 22))
MIN_BLOB = int(os.environ.get("MIN_BLOB", 700))
GLASSES = os.environ.get("GLASSES", "0") != "0"
EYE_SHARP = os.environ.get("EYE_SHARP", "0") != "0"
FILL_BG = os.environ.get("FILL_BG", "1") != "0"   # render background as dark pixels
BG_PIX = float(os.environ.get("BG_PIX", 0.07))    # background pixel brightness
# glasses placement (fractions of the face bbox) -- tune these from the preview
EYE_ROW = float(os.environ.get("EYE_ROW", 0.46))
EYE_HALF = float(os.environ.get("EYE_HALF", 0.20))
LENS_RX = float(os.environ.get("LENS_RX", 0.15))
LENS_RY = float(os.environ.get("LENS_RY", 0.11))

# ---- load + crop ----
src = Image.open(INP).convert("RGB")
w, h = src.size
side = int(min(w, h) * CROP_SCALE)
left = (w - side) // 2
top = max(0, min(int((h - side) * TOP_BIAS), h - side))
src = src.crop((left, top, left + side, top + side))
W, H = src.size
gray = src.convert("L")

# ---- background flood-fill -> mask ----
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

subj = ~bg
labels = np.zeros((H, W), np.int32)
sizes = [0]; cur = 0
for y0 in range(H):
    for x0 in range(W):
        if subj[y0, x0] and labels[y0, x0] == 0:
            cur += 1; q = deque([(y0, x0)]); labels[y0, x0] = cur; n = 0
            while q:
                y, x = q.popleft(); n += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and subj[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = cur; q.append((ny, nx))
            sizes.append(n)
keep = np.array([s >= MIN_BLOB for s in sizes])
subj = subj & keep[labels] if cur else subj

# ---- grayscale enhance + downsample ----
en = ImageEnhance.Contrast(gray).enhance(CONTRAST)
en = en.filter(ImageFilter.UnsharpMask(radius=2, percent=int(SHARP * 100), threshold=1))
ROWS = round(COLS * H / W)
lum = np.asarray(en.resize((COLS, ROWS), Image.LANCZOS), np.float32) / 255.0
mask = np.asarray(Image.fromarray((subj * 255).astype(np.uint8))
                  .resize((COLS, ROWS), Image.BILINEAR)) > 110

# quantise to LEVELS grey steps for a crisp block look
q = np.round(lum * (LEVELS - 1)) / (LEVELS - 1)

# ---- face bbox + eye anchors ----
ys, xs = np.where(mask)
r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
fh, fw = r1 - r0, c1 - c0

# head width ~ mask width across the forehead; head is roughly square, so the
# chin sits about one head-width below the hairline -> excludes the shirt.
fore_r = r0 + int(0.18 * fh)
fore_cols = np.where(mask[min(fore_r, ROWS - 1)])[0]
head_w = (fore_cols.max() - fore_cols.min()) if len(fore_cols) else fw
head_bot = min(r1, r0 + int(1.35 * head_w))
hh = head_bot - r0                      # head height (face only)

if os.environ.get("EYE_ROW_ABS"):
    eye_y = int(os.environ["EYE_ROW_ABS"])
else:
    # strongest dark horizontal band in the upper face = brows; eyes just below
    lo_r = r0 + int(0.28 * hh); hi_r = r0 + int(0.62 * hh)
    dcount = [int((mask[r] & (q[r] < 0.33)).sum()) for r in range(lo_r, hi_r)]
    brow = lo_r + (int(np.argmax(dcount)) if dcount else int(0.1 * hh))
    eye_y = min(ROWS - 2, brow + max(1, int(0.05 * hh)))

# eye columns: darkest cluster each side of the head centre at the eye row
band = mask[max(0, eye_y - 1):eye_y + 2].any(axis=0)
hcols = np.where(mask[eye_y])[0]
cc = int(hcols.mean()) if len(hcols) else (c0 + c1) // 2
def eye_col(side):
    rng = range(c0, cc) if side < 0 else range(cc, c1)
    best, bv = cc + side * int(0.2 * head_w), 9
    for x in rng:
        if band[x]:
            v = min(q[r][x] for r in range(max(0, eye_y - 1), eye_y + 2))
            if v < bv:
                bv, best = v, x
    return best
# Eye centres (col,row) and lens radius, hand-tuned for this photo at COLS=82.
# Override with EYES="cxL,cyL,cxR,cyR" / LENS_ABS="rx,ry", or EYES=auto to detect.
EYES = os.environ.get("EYES", "33,28,48,28")
LENS_DEF = os.environ.get("LENS_ABS", "6,6")
if EYES == "auto":
    eyes = [(eye_col(-1), eye_y), (eye_col(1), eye_y)]
else:
    a, b, c, d = (int(v) for v in EYES.split(","))
    eyes = [(a, b), (c, d)]; eye_y = b
rx, ry = (int(v) for v in LENS_DEF.split(","))

FRAME = 1.0     # glasses frame = brightest ink
DARK = 0.06     # sharpened pupil

def sharpen_eye(ex, ey):
    for yy in range(ey - 1, ey + 2):
        for xx in range(ex - 1, ex + 2):
            if 0 <= yy < ROWS and 0 <= xx < COLS and mask[yy, xx]:
                q[yy, xx] = DARK
    if 0 <= ey < ROWS and 0 <= ex < COLS:
        q[ey, ex] = 0.9     # single catch-light pixel

glass_cells = set()
def lens(ex, ey):
    # elliptical ring of frame pixels around the eye
    for a in range(0, 360, 6):
        import math
        xx = int(round(ex + rx * math.cos(math.radians(a))))
        yy = int(round(ey + ry * math.sin(math.radians(a))))
        if 0 <= yy < ROWS and 0 <= xx < COLS:
            glass_cells.add((yy, xx))

if GLASSES:
    for (ex, ey) in eyes:
        if EYE_SHARP:
            sharpen_eye(ex, ey)
        lens(ex, ey)
    # bridge between the two lenses
    by = eye_y
    for xx in range(eyes[0][0] + rx, eyes[1][0] - rx + 1):
        glass_cells.add((by, xx)); glass_cells.add((by - 1, xx))
    # temple arms out to the sides
    for xx in range(max(c0, eyes[0][0] - rx - 5), eyes[0][0] - rx + 1):
        glass_cells.add((eye_y - 1, xx))
    for xx in range(eyes[1][0] + rx, min(c1, eyes[1][0] + rx + 6) + 1):
        glass_cells.add((eye_y - 1, xx))
elif EYE_SHARP:
    for (ex, ey) in eyes:
        sharpen_eye(ex, ey)

print("eyes at", eyes, "| lens", (rx, ry), "| face bbox rows", (int(r0), int(r1)))

# ---- render SVG ----
BG = "#0d1117"; BG2 = "#111722"; FRAMEC = "#30363d"; TITLE = "#7d8590"; INK = "#c9d1d9"
PAD = 20; TITLEBAR_H = 30; STATUS_H = 30
ART_W = 800; CELL = ART_W / COLS; ART_H = CELL * ROWS
CANVAS_W = ART_W + PAD * 2; CANVAS_H = TITLEBAR_H + ART_H + STATUS_H + PAD
art_top = TITLEBAR_H + PAD * 0.35
REVEAL = 3.2; rows_delay = REVEAL / ROWS

def shade(v):  # mix dark bg -> light ink by v, return hex
    b = (13, 17, 23); k = (201, 209, 217)
    return "#%02x%02x%02x" % tuple(int(b[i] + (k[i] - b[i]) * v) for i in range(3))

p = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W:.0f}" height="{CANVAS_H:.0f}" '
    f'viewBox="0 0 {CANVAS_W:.0f} {CANVAS_H:.0f}" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">',
    '<style>.rw{opacity:0;animation:pop .45s ease-out both}'
    '@keyframes pop{0%{opacity:0;transform:translateY(-6px)}100%{opacity:1;transform:translateY(0)}}'
    '@media (prefers-reduced-motion:reduce){.rw{opacity:1!important;transform:none!important;animation:none!important}}</style>',
    f'<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">'
    f'<stop offset="0" stop-color="{BG2}"/><stop offset="1" stop-color="{BG}"/></linearGradient></defs>',
    f'<rect width="{CANVAS_W:.0f}" height="{CANVAS_H:.0f}" rx="12" fill="url(#bg)"/>',
    f'<rect x="0.5" y="0.5" width="{CANVAS_W-1:.0f}" height="{CANVAS_H-1:.0f}" rx="12" fill="none" stroke="{FRAMEC}"/>',
    f'<line x1="0" y1="{TITLEBAR_H}" x2="{CANVAS_W:.0f}" y2="{TITLEBAR_H}" stroke="{FRAMEC}"/>',
]
for i, dot in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
    p.append(f'<circle cx="{PAD + i*16}" cy="{TITLEBAR_H/2}" r="5" fill="{dot}"/>')
p.append(f'<text x="{CANVAS_W/2:.0f}" y="{TITLEBAR_H/2 + 4:.0f}" fill="{TITLE}" font-size="12" '
         f'text-anchor="middle">kunal@github: ~$ ./portrait.sh</text>')

gap = 0.5
for ry_ in range(ROWS):
    cells = []
    for cx in range(COLS):
        on = mask[ry_, cx]
        is_frame = (ry_, cx) in glass_cells
        if is_frame:
            v = FRAME
        elif on:
            v = q[ry_, cx]
        elif FILL_BG:
            v = BG_PIX                     # background rendered as a dark pixel
        else:
            continue
        x = PAD + cx * CELL; y = art_top + ry_ * CELL
        cells.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{CELL-gap:.1f}" height="{CELL-gap:.1f}" '
                     f'rx="1" fill="{shade(v)}"/>')
    if cells:
        p.append(f'<g class="rw" style="animation-delay:{ry_*rows_delay:.2f}s">{"".join(cells)}</g>')

sly = TITLEBAR_H + ART_H + PAD * 0.35; sy = sly + 19
p.append(f'<line x1="0" y1="{sly:.1f}" x2="{CANVAS_W:.0f}" y2="{sly:.1f}" stroke="{FRAMEC}"/>')
p.append(f'<text x="{PAD}" y="{sy:.1f}" fill="{TITLE}" font-size="13">'
         f'kunal@github:~$ whoami <tspan fill="{INK}">Kunal Kumar</tspan></text>')
p.append("</svg>")
svg = "".join(p)
open(OUT, "w", encoding="utf-8").write(svg)
print("wrote", OUT, len(svg) // 1024, "KB;", f"{COLS}x{ROWS} grid")

if PNG:
    scale = 10; sc = scale / 2
    pw, ph = int(CANVAS_W * sc), int(CANVAS_H * sc)
    prev = Image.new("RGB", (pw, ph), (13, 17, 23)); d = ImageDraw.Draw(prev)
    for ry_ in range(ROWS):
        for cx in range(COLS):
            on = mask[ry_, cx]; is_frame = (ry_, cx) in glass_cells
            if is_frame:
                v = FRAME
            elif on:
                v = q[ry_, cx]
            elif FILL_BG:
                v = BG_PIX
            else:
                continue
            x = (PAD + cx * CELL) * sc; y = (art_top + ry_ * CELL) * sc
            col = tuple(int(13 + (c2 - 13) * v) if i == 0 else
                        int((17 if i == 1 else 23) + ((209 if i == 1 else 217) - (17 if i == 1 else 23)) * v)
                        for i, c2 in enumerate((201, 209, 217)))
            d.rectangle([x, y, x + (CELL - gap) * sc, y + (CELL - gap) * sc], fill=col)
    prev.save(PNG); print("wrote preview", PNG, prev.size)
