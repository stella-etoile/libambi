# ambi/refinement.py
from __future__ import annotations
import numpy as np
from typing import Tuple

__all__ = ["pack_residual", "apply_residual"]

def _cell_splits(n: int, g: int):
    """Split a length n into g nearly-equal segments; returns boundaries [0, ..., n]."""
    # Example: n=33,g=4 -> [0, 8, 17, 25, 33]
    base = n // g
    extra = n % g
    sizes = [base + (1 if i < extra else 0) for i in range(g)]
    bounds = [0]
    acc = 0
    for s in sizes:
        acc += s
        bounds.append(acc)
    return bounds

def pack_residual(
    truth: np.ndarray,
    base: np.ndarray,
    q_step: float = 0.02,
    grid: int = 4,
    max_mag: int = 4,
) -> bytes:
    """
    Build a tiny residual for (truth - base) as a g×g×3 grid of per-cell mean deltas, quantized to int8.
    Returns bytes: [1-byte g][g*g*3 int8 values].
      - g in [2..8] typically; auto-reduced for small blocks.
      - q_step: dequant step (same used on decode).
      - max_mag: clamp range (int8 in [-max_mag, +max_mag]).
    If block is too small (g<2) or residual is negligible, returns b"".
    """
    H, W, C = truth.shape
    assert C == 3
    g = int(grid)
    # auto-reduce for small blocks; need at least 2×2 cells and cell ≥ 2×2
    g = max(2, min(g, max(2, min(H // 2, W // 2))))
    if g < 2:
        return b""

    delta = (truth.astype(np.float32) - base.astype(np.float32))
    yb = _cell_splits(H, g)
    xb = _cell_splits(W, g)

    cell_means = np.zeros((g, g, 3), dtype=np.float32)
    for yi in range(g):
        y0, y1 = yb[yi], yb[yi + 1]
        for xi in range(g):
            x0, x1 = xb[xi], xb[xi + 1]
            cell = delta[y0:y1, x0:x1, :]
            if cell.size == 0:
                continue
            cell_means[yi, xi, :] = cell.mean(axis=(0, 1))

    # quantize to int8 with symmetric clamp
    if q_step <= 0:
        q_step = 0.02
    max_q = int(max_mag)
    q = np.clip(np.round(cell_means / q_step), -max_q, max_q).astype(np.int8)

    # early-out if almost all zeros (no gain)
    if np.count_nonzero(q) == 0:
        return b""

    # bytes: [g (uint8)] + raw int8 grid (g*g*3)
    g_byte = bytes([g & 0xFF])
    return g_byte + q.tobytes(order="C")


def apply_residual(
    base: np.ndarray,
    payload: bytes,
    q_step: float = 0.02,
) -> np.ndarray:
    """
    Apply a packed residual payload onto 'base' and return the refined block.
    Payload format: [1-byte g][g*g*3 int8 values].
    We upsample by nearest-neighbor per cell to match base H×W×3.
    """
    if not payload:
        return base
    H, W, C = base.shape
    assert C == 3

    g = payload[0]
    vals = np.frombuffer(payload, dtype=np.int8, offset=1)
    if vals.size != g * g * 3:
        # corrupted payload — ignore residual
        return base

    grid = vals.reshape((g, g, 3)).astype(np.float32) * float(q_step)

    # expand grid to H×W by repeating per-cell
    yb = _cell_splits(H, g)
    xb = _cell_splits(W, g)
    up = np.zeros_like(base, dtype=np.float32)
    for yi in range(g):
        y0, y1 = yb[yi], yb[yi + 1]
        for xi in range(g):
            x0, x1 = xb[xi], xb[xi + 1]
            up[y0:y1, x0:x1, :] = grid[yi, xi, :]

    out = base.astype(np.float32) + up
    # clamp to valid range [0,1] — our pipeline uses floats in [0,1] for YCbCr planes
    np.clip(out, 0.0, 1.0, out=out)
    return out