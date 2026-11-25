from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

_FAMILY_COLORS = {"dct": (40, 120, 255), "flat": (40, 200, 90), "ts": (255, 90, 60), "intra": (255, 190, 40)}


def draw_partition(rgb: np.ndarray, leaves: list, cfg, color_by_mode: bool = True,
                   scale: int = 2) -> Image.Image:
    """Overlay leaf rectangles on an RGB image. `leaves` are in image pixel coords."""
    im = Image.fromarray(rgb).resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.NEAREST)
    d = ImageDraw.Draw(im, "RGBA")
    for l in leaves:
        col = _FAMILY_COLORS[cfg.mode_family(l.mode)] if color_by_mode else (255, 255, 255)
        d.rectangle([l.x * scale, l.y * scale, (l.x + l.w) * scale - 1, (l.y + l.h) * scale - 1],
                    outline=col + (230,), fill=col + (28,) if color_by_mode else None)
    return im
