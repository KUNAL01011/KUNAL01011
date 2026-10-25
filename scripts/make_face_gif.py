"""
Animated head-shot GIF from the original photo: circular head/face crop on the
dark terminal background, round glasses drawn on, with a gentle breathing zoom,
a glint that sweeps the lenses, and a soft scan line. Pillow only.

    python scripts/make_face_gif.py [input] [output.gif]

Env: SIZE, FRAMES, CX, CY, RAD, EXL, EXR, EYY, LENSR, CROP_SCALE, TOP_BIAS.
"""
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
INP = args[0] if len(args) > 0 else os.path.join(HERE, "..", "source-photo.jpg")
OUT = args[1] if len(args) > 1 else os.path.join(HERE, "..", "kunal-face.gif")
DUMP = sys.argv[sys.argv.index("--dump") + 1] if "--dump" in sys.argv else None

S = int(os.environ.get("SIZE", 440))
FRAMES = int(os.environ.get("FRAMES", 36))
CROP_SCALE = float(os.environ.get("CROP_SCALE", 0.84))
TOP_BIAS = float(os.environ.get("TOP_BIAS", 0.18))
# circle (head) + eyes, as fractions of S (match the pixel portrait's framing)
CX = float(os.environ.get("CX", 0.50)); CY = float(os.environ.get("CY", 0.40))
RAD = float(os.environ.get("RAD", 0.47))
EXL = float(os.environ.get("EXL", 33.5 / 82)); EXR = float(os.environ.get("EXR", 48.5 / 82))
EYY = float(os.environ.get("EYY", 28.5 / 82)); LENSR = float(os.environ.get("LENSR", 7.2 / 82))

BG = (13, 17, 23)
ACCENT = (34, 211, 238)   # cyan terminal accent

# ---- load + square crop (same framing as the other scripts) ----
src = Image.open(INP).convert("RGB")
w, h = src.size
side = int(min(w, h) * CROP_SCALE)
left = (w - side) // 2
top = max(0, min(int((h - side) * TOP_BIAS), h - side))
base = src.crop((left, top, left + side, top + side)).resize((S, S), Image.LANCZOS)

cx, cy, R = CX * S, CY * S, RAD * S
exl, exr, eyy = EXL * S, EXR * S, EYY * S
lr = LENSR * S

# circular head mask (feathered)
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).ellipse([cx - R, cy - R, cx + R, cy + R], fill=255)
mask = mask.filter(ImageFilter.GaussianBlur(S * 0.012))

bg = Image.new("RGB", (S, S), BG)


def draw_glasses(layer, glint_t=None):
    d = ImageDraw.Draw(layer, "RGBA")
    fw = max(3, int(lr * 0.20))                       # frame thickness
    tint = (10, 12, 16, 70)                            # faint lens tint
    for ex in (exl, exr):
        d.ellipse([ex - lr, eyy - lr, ex + lr, eyy + lr], fill=tint)
        d.ellipse([ex - lr, eyy - lr, ex + lr, eyy + lr], outline=(18, 20, 24, 255), width=fw)
        d.ellipse([ex - lr, eyy - lr, ex + lr, eyy + lr], outline=(60, 65, 72, 160), width=max(1, fw // 2))
    # bridge
    d.line([(exl + lr, eyy), (exr - lr, eyy)], fill=(18, 20, 24, 255), width=fw)
    # temple arms out to the circle edge
    d.line([(exl - lr, eyy - 2), (cx - R * 0.98, eyy - lr * 0.4)], fill=(18, 20, 24, 255), width=fw)
    d.line([(exr + lr, eyy - 2), (cx + R * 0.98, eyy - lr * 0.4)], fill=(18, 20, 24, 255), width=fw)
    # glint: a bright diagonal streak crossing each lens
    if glint_t is not None:
        gl = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(gl)
        for ex in (exl, exr):
            off = (glint_t * 2 - 1) * lr * 1.4
            gx = ex + off
            gd.line([(gx - lr * 0.5, eyy - lr), (gx + lr * 0.5, eyy + lr)],
                    fill=(255, 255, 255, 150), width=max(2, int(lr * 0.22)))
        # clip glint to the two lenses
        clip = Image.new("L", layer.size, 0)
        cd = ImageDraw.Draw(clip)
        for ex in (exl, exr):
            cd.ellipse([ex - lr, eyy - lr, ex + lr, eyy + lr], fill=255)
        layer.paste(Image.alpha_composite(layer.convert("RGBA"),
                    Image.composite(gl, Image.new("RGBA", layer.size, (0, 0, 0, 0)), clip)), (0, 0))


frames = []
for f in range(FRAMES):
    t = f / FRAMES
    # breathing zoom
    z = 1.0 + 0.035 * math.sin(2 * math.pi * t)
    zs = int(S * z)
    zi = base.resize((zs, zs), Image.LANCZOS)
    ox, oy = (zs - S) // 2, int((zs - S) * 0.42)
    zi = zi.crop((ox, oy, ox + S, oy + S))

    frame = Image.composite(zi, bg, mask).convert("RGBA")

    # glasses (+ glint during the first ~40% of the loop)
    glint = (t / 0.4) if t < 0.4 else None
    draw_glasses(frame, glint)

    # accent ring
    ring = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(ring).ellipse([cx - R, cy - R, cx + R, cy + R],
                                 outline=ACCENT + (120,), width=3)
    frame = Image.alpha_composite(frame, ring)

    # soft scan line sweeping down once per loop
    sy = int((t * 1.2 - 0.1) * S)
    if 0 <= sy < S:
        sl = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        ImageDraw.Draw(sl).rectangle([0, sy, S, sy + 2], fill=ACCENT + (40,))
        frame = Image.alpha_composite(frame, Image.composite(
            sl, Image.new("RGBA", (S, S), (0, 0, 0, 0)), mask))

    frames.append(frame.convert("RGB"))

frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=70, loop=0, optimize=True)
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB;", FRAMES, "frames", f"{S}x{S}")

if DUMP:
    frames[0].save(DUMP + "_f0.png")
    frames[FRAMES // 5].save(DUMP + "_fg.png")   # during glint
    frames[FRAMES // 2].save(DUMP + "_fm.png")
    print("dumped frames")
