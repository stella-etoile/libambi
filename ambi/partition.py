"""
Quadtree Keep/Split (heuristic) for AMBI.

Exports:
- dynamic_quadtree_leaves(ycc, min_block, max_block, var_thresh=None, grad_thresh=None)
  -> list[(x, y, w, h)] in raster order.

Thresholds are on the luma (Y) plane in [0, 1].
If var_thresh/grad_thresh are not provided, we fall back to env vars, then defaults:
  AMBI_SPLIT_VAR  (default 0.0025)
  AMBI_SPLIT_GRAD (default 0.020)
"""
from __future__ import annotations
import os
import numpy as np
from typing import List, Tuple

__all__ = ["dynamic_quadtree_leaves"]

def _grad_mean_abs(y: np.ndarray) -> float:
    """Mean absolute gradient magnitude (very lightweight)."""
    if y.size == 0:
        return 0.0
    yf = y.astype(np.float32, copy=False)
    gx = np.abs(np.diff(yf, axis=1))
    gy = np.abs(np.diff(yf, axis=0))
    h = min(gx.shape[0], gy.shape[0])
    w = min(gx.shape[1], gy.shape[1])
    if h <= 0 or w <= 0:
        return 0.0
    g = gx[:h, :w] + gy[:h, :w]
    return float(g.mean())

def _should_split_block(yplane: np.ndarray, x: int, y: int, w: int, h: int,
                        min_block: int,
                        var_thresh: float, grad_thresh: float) -> bool:
    """Decision = (var >= var_thresh) OR (mean|grad| >= grad_thresh)."""
    if w <= min_block or h <= min_block:
        return False
    sub = yplane[y:y+h, x:x+w]
    if sub.size == 0:
        return False
    # luma variance
    if float(sub.var()) >= var_thresh:
        return True
    # simple gradient magnitude
    return _grad_mean_abs(sub) >= grad_thresh

def _split_tile_rec(yplane: np.ndarray, x: int, y: int, w: int, h: int,
                    min_block: int, var_thresh: float, grad_thresh: float,
                    out_leaves: list) -> None:
    """Recursive TL, TR, BL, BR split; handles edge slivers gracefully."""
    # stop if already at (or below) min
    if w <= min_block and h <= min_block:
        out_leaves.append((x, y, w, h))
        return

    # one-axis split if only one dimension is splittable
    if (w >= 2 * min_block and h < 2 * min_block):
        w2 = w // 2
        _split_tile_rec(yplane, x, y, w2, h, min_block, var_thresh, grad_thresh, out_leaves)
        _split_tile_rec(yplane, x + w2, y, w - w2, h, min_block, var_thresh, grad_thresh, out_leaves)
        return
    if (h >= 2 * min_block and w < 2 * min_block):
        h2 = h // 2
        _split_tile_rec(yplane, x, y, w, h2, min_block, var_thresh, grad_thresh, out_leaves)
        _split_tile_rec(yplane, x, y + h2, w, h - h2, min_block, var_thresh, grad_thresh, out_leaves)
        return

    # 2-D split decision
    if not _should_split_block(yplane, x, y, w, h, min_block, var_thresh, grad_thresh):
        out_leaves.append((x, y, w, h))
        return

    # split into 4 (handle odd sizes at borders)
    w2 = w // 2
    h2 = h // 2
    # TL
    _split_tile_rec(yplane, x, y, w2, h2, min_block, var_thresh, grad_thresh, out_leaves)
    # TR
    _split_tile_rec(yplane, x + w2, y, w - w2, h2, min_block, var_thresh, grad_thresh, out_leaves)
    # BL
    _split_tile_rec(yplane, x, y + h2, w2, h - h2, min_block, var_thresh, grad_thresh, out_leaves)
    # BR
    _split_tile_rec(yplane, x + w2, y + h2, w - w2, h - h2, min_block, var_thresh, grad_thresh, out_leaves)

def dynamic_quadtree_leaves(ycc: np.ndarray, min_block: int, max_block: int,
                            var_thresh: float | None = None,
                            grad_thresh: float | None = None) -> List[Tuple[int, int, int, int]]:
    """
    Tile the image with max_block cells, then recursively Keep/Split each cell.
    Returns raster-ordered leaf rectangles (x, y, w, h).
    """
    h, w = ycc.shape[:2]
    yplane = ycc[..., 0]  # luma in [0,1]

    if var_thresh is None:
        var_thresh = float(os.environ.get("AMBI_SPLIT_VAR", "0.0025"))
    if grad_thresh is None:
        grad_thresh = float(os.environ.get("AMBI_SPLIT_GRAD", "0.020"))

    leaves: list[tuple[int, int, int, int]] = []
    for yy in range(0, h, max_block):
        bh = min(max_block, h - yy)
        for xx in range(0, w, max_block):
            bw = min(max_block, w - xx)
            _split_tile_rec(yplane, xx, yy, bw, bh, min_block, var_thresh, grad_thresh, leaves)
    return leaves