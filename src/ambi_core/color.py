from __future__ import annotations

import numpy as np


def rgb_to_ycc(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=0).astype(np.float32)


def ycc_to_rgb(ycc: np.ndarray) -> np.ndarray:
    y, cb, cr = ycc[0], ycc[1] - 128.0, ycc[2] - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    out = np.stack([r, g, b], axis=-1)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def psnr_rgb(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0**2 / mse))
