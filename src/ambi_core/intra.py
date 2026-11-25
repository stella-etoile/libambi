from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import INTRA_PRED_NAMES

_SLOPES = {"drv": (1, 2), "dr": (1, 1), "drh": (2, 1)}


def new_ext(height: int, width: int) -> np.ndarray:
    return np.full((3, height + 1, width + 1), np.nan, dtype=np.float32)


def gather_refs(ext: np.ndarray, ys, xs, h: int, w: int):
    ys = np.asarray(ys, dtype=np.int64)
    xs = np.asarray(xs, dtype=np.int64)
    above = ext[:, ys[:, None], xs[:, None] + 1 + np.arange(w)[None, :]]
    left = ext[:, ys[:, None] + 1 + np.arange(h)[None, :], xs[:, None]]
    corner = ext[:, ys, xs]
    return above.transpose(1, 0, 2), left.transpose(1, 0, 2), corner.T


@lru_cache(maxsize=None)
def _diag_index(kind: str, h: int, w: int) -> np.ndarray:
    num, den = _SLOPES[kind]
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    if (num, den) == (1, 2):
        shift_row, shift_col = (y + 2) // 2, 2 * (x + 1)
    elif (num, den) == (1, 1):
        shift_row, shift_col = y + 1, x + 1
    else:
        shift_row, shift_col = 2 * (y + 1), (x + 2) // 2
    ax = x - shift_row
    ly = np.maximum(y - shift_col, -1)
    from_above = ax >= -1
    idx_above = h + 1 + ax
    idx_left = np.where(ly >= 0, h - 1 - ly, h)
    return np.where(from_above, idx_above, idx_left).astype(np.int64)


def substitute(above, left, corner):
    have_a = ~np.isnan(above[:, 0, 0])
    have_l = ~np.isnan(left[:, 0, 0])
    a3, l3 = have_a[:, None, None], have_l[:, None, None]
    A = np.where(a3, above, np.where(l3, left[:, :, :1], 128.0))
    L = np.where(l3, left, np.where(a3, above[:, :, :1], 128.0))
    both = (have_a & have_l)[:, None]
    C = np.where(both, corner, np.where(have_a[:, None], above[:, :, 0],
                                        np.where(have_l[:, None], left[:, :, 0], 128.0)))
    return A.astype(np.float32), L.astype(np.float32), C.astype(np.float32), have_a, have_l


def predict(kind: str, above, left, corner, h: int, w: int) -> np.ndarray:
    n = above.shape[0]
    A, L, C, have_a, have_l = substitute(above, left, corner)
    if kind == "dc":
        sa = np.nan_to_num(above, nan=0.0).astype(np.int64).sum(axis=2)
        sl = np.nan_to_num(left, nan=0.0).astype(np.int64).sum(axis=2)
        both = (have_a & have_l)[:, None]
        dc = np.where(both, (sa + sl + (w + h) // 2) // (w + h),
                      np.where(have_a[:, None], (sa + w // 2) // w,
                               np.where(have_l[:, None], (sl + h // 2) // h, 128)))
        return np.broadcast_to(dc.astype(np.float32)[:, :, None, None], (n, 3, h, w)).copy()
    if kind == "v":
        return np.broadcast_to(A[:, :, None, :], (n, 3, h, w)).copy()
    if kind == "h":
        return np.broadcast_to(L[:, :, :, None], (n, 3, h, w)).copy()
    if kind == "tm":
        p = L[:, :, :, None].astype(np.int64) + A[:, :, None, :].astype(np.int64) - C[:, :, None, None].astype(np.int64)
        return np.clip(p, 0, 255).astype(np.float32)
    if kind in _SLOPES:
        ref = np.concatenate([L[:, :, ::-1], C[:, :, None], A], axis=2)
        return ref[:, :, _diag_index(kind, h, w)]
    raise ValueError(f"unknown intra predictor {kind!r}")


def predict_all(preds: tuple, ext: np.ndarray, ys, xs, h: int, w: int) -> list:
    above, left, corner = gather_refs(ext, ys, xs, h, w)
    return [predict(k, above, left, corner, h, w) for k in preds]
