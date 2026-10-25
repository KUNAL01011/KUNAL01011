"""
Compose an ORIGINAL armored-warrior scene (my own design, not based on any
existing character): a dark knight holding an upright greatsword, wreathed in
blue energy flames on black. Rendered with Pillow + numpy primitives, meant to
be pixelated afterwards by make_pixel_svg.py.

    python scripts/make_warrior.py [output.png]
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "warrior-src.png")
S = 600
cx = S // 2

# palette
BLACK = (6, 9, 15)
STEEL0 = (20, 25, 35)      # deep shadow armor
STEEL1 = (41, 50, 66)      # mid armor
STEEL2 = (78, 96, 122)     # armor highlight
RIM = (120, 200, 255)      # blue rim light
BLUE = np.array([30, 120, 220], float)
CYAN = np.array([120, 215, 255], float)
WHITE = np.array([230, 248, 255], float)

rng = np.random.default_rng(7)

# ---- 1. blue energy flames (procedural) behind the figure ----
def flame_field():
    base = rng.random((S // 4, S // 4))
    f = Image.fromarray((base * 255).astype(np.uint8)).resize((S, S), Image.BICUBIC)
    f = f.filter(ImageFilter.GaussianBlur(10))
    n = np.asarray(f, float) / 255.0
    n2 = np.asarray(Image.fromarray((rng.random((S // 8, S // 8)) * 255).astype(np.uint8))
                    .resize((S, S), Image.BICUBIC).filter(ImageFilter.GaussianBlur(6)), float) / 255.0
    turb = (n * 0.6 + n2 * 0.4)
    yy = np.linspace(1.0, 0.0, S)[:, None]          # stronger low, fades up
    xx = np.linspace(0, 1, S)[None, :]
    side = 0.5 + 0.9 * xx                            # brighter toward the right
    rising = np.clip(turb ** 2.2 * (0.35 + 0.9 * yy) * side, 0, 1)
    rising = np.asarray(Image.fromarray((rising * 255).astype(np.uint8))
                        .filter(ImageFilter.GaussianBlur(3)), float) / 255.0
    return np.clip(rising * 1.5, 0, 1)

intensity = flame_field()
img = np.zeros((S, S, 3), float)
img[:] = BLACK
for i in range(3):
    t = intensity
    col = (BLUE * (1 - np.clip(t * 1.6, 0, 1))[..., None]
           + CYAN * np.clip(t * 1.6, 0, 1)[..., None])
    col = col * (0.4 + 0.6 * t)[..., None]
    core = np.clip((t - 0.72) / 0.28, 0, 1)
    col = col * (1 - core)[..., None] + WHITE * core[..., None]
    img[..., i] = np.maximum(img[..., i], col[..., i])
scene = Image.fromarray(img.astype(np.uint8), "RGB")
d = ImageDraw.Draw(scene, "RGBA")

# ---- 2. cape (behind the body) ----
d.polygon([(cx - 70, 150), (cx + 70, 150), (cx + 150, 560), (cx - 150, 560)],
          fill=(10, 13, 20, 255))
d.polygon([(cx - 60, 160), (cx + 60, 160), (cx + 120, 555), (cx, 520), (cx - 120, 555)],
          fill=(16, 20, 30, 255))

# ---- 3. greatsword (upright, held in front) ----
blade_w = 46
d.polygon([(cx - blade_w // 2, 70), (cx + blade_w // 2, 70),
           (cx + blade_w // 2, 250), (cx, 300), (cx - blade_w // 2, 250)],
          fill=(150, 170, 195, 255))                                  # blade
d.polygon([(cx - blade_w // 2, 70), (cx, 300), (cx, 70)], fill=(95, 115, 140, 255))  # bevel
d.line([(cx, 60), (cx, 300)], fill=(210, 235, 255, 180), width=3)     # fuller glint
d.rectangle([cx - 95, 300, cx + 95, 322], fill=(60, 72, 92, 255))     # crossguard
d.rectangle([cx - 95, 300, cx + 95, 322], outline=RIM + (200,), width=2)
d.rectangle([cx - 12, 322, cx + 12, 378], fill=(34, 40, 52, 255))     # grip
d.ellipse([cx - 16, 374, cx + 16, 404], fill=(70, 84, 104, 255))      # pommel

# ---- 4. warrior body (original design) ----
def plate(poly, base=STEEL1, hi=STEEL2, rim=True):
    d.polygon(poly, fill=base + (255,))
    if rim:
        d.line(poly + [poly[0]], fill=RIM + (140,), width=2)

# legs / greaves
plate([(cx - 46, 378), (cx - 8, 378), (cx - 14, 560), (cx - 52, 560)], STEEL0, STEEL1)
plate([(cx + 8, 378), (cx + 46, 378), (cx + 52, 560), (cx + 14, 560)], STEEL0, STEEL1)
# faulds / belt
plate([(cx - 60, 338), (cx + 60, 338), (cx + 50, 392), (cx - 50, 392)], STEEL1, STEEL2)
# cuirass (chest)
plate([(cx - 70, 196), (cx + 70, 196), (cx + 60, 344), (cx - 60, 344)], STEEL1, STEEL2)
d.line([(cx, 200), (cx, 340)], fill=(10, 13, 20, 220), width=3)        # center seam
d.polygon([(cx - 70, 196), (cx, 250), (cx + 70, 196)], fill=(10, 13, 20, 120))  # collar shadow
# pauldrons (shoulders)
d.ellipse([cx - 104, 176, cx - 44, 236], fill=STEEL2 + (255,))
d.ellipse([cx + 44, 176, cx + 104, 236], fill=STEEL2 + (255,))
d.ellipse([cx - 100, 180, cx - 48, 232], outline=RIM + (150,), width=2)
d.ellipse([cx + 48, 180, cx + 100, 232], outline=RIM + (150,), width=2)
# arms down to the hilt (both hands grip the sword)
plate([(cx - 96, 206), (cx - 60, 214), (cx - 20, 316), (cx - 44, 330)], STEEL0, STEEL1)
plate([(cx + 96, 206), (cx + 60, 214), (cx + 20, 316), (cx + 44, 330)], STEEL0, STEEL1)
# gauntlets on the grip
d.ellipse([cx - 30, 308, cx + 2, 346], fill=STEEL2 + (255,))
d.ellipse([cx - 2, 308, cx + 30, 346], fill=STEEL1 + (255,))
# head: helmet with a crest fin (original)
d.ellipse([cx - 30, 96, cx + 30, 170], fill=STEEL1 + (255,))           # helm
d.ellipse([cx - 30, 96, cx + 30, 170], outline=RIM + (160,), width=2)
d.polygon([(cx - 4, 70), (cx + 4, 70), (cx + 10, 150), (cx - 10, 150)], fill=STEEL2 + (255,))  # crest
d.polygon([(cx - 26, 128), (cx + 26, 128), (cx + 20, 150), (cx - 20, 150)], fill=(8, 10, 16, 255))  # visor
d.line([(cx - 22, 138), (cx + 22, 138)], fill=(90, 200, 255, 230), width=3)  # glowing visor slit

# ---- 5. blue rim glow on the whole figure ----
glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
# re-draw a soft blue halo behind the silhouette
halo = Image.new("L", (S, S), 0)
hd = ImageDraw.Draw(halo)
hd.ellipse([cx - 150, 80, cx + 150, 560], fill=120)
halo = halo.filter(ImageFilter.GaussianBlur(40))
blue_layer = Image.new("RGBA", (S, S), RIM + (0,))
blue_layer.putalpha(halo.point(lambda v: int(v * 0.5)))
scene = Image.alpha_composite(scene.convert("RGBA"), blue_layer).convert("RGB")
scene.paste(Image.alpha_composite(scene.convert("RGBA"), glow).convert("RGB"), (0, 0))

scene.save(OUT)
print("wrote", OUT, scene.size)
