from __future__ import annotations

import numpy as np

from .color import rgb_to_ycc
from .distortion import C1, C2, box_mean


def ssim_y(a: np.ndarray, b: np.ndarray, win: int = 7) -> float:
    ya = rgb_to_ycc(a)[0][None] / 255.0
    yb = rgb_to_ycc(b)[0][None] / 255.0
    win = max(1, min(win, ya.shape[1], ya.shape[2]))
    if win % 2 == 0:
        win -= 1
    mx, my = box_mean(ya, win), box_mean(yb, win)
    vx = box_mean(ya * ya, win) - mx**2
    vy = box_mean(yb * yb, win) - my**2
    cov = box_mean(ya * yb, win) - mx * my
    s = ((2 * mx * my + C1) * (2 * cov + C2)) / ((mx**2 + my**2 + C1) * (vx + vy + C2))
    return float(s.mean())
