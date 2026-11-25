from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def _dct_matrix64(n: int) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)[:, None]
    i = np.arange(n, dtype=np.float64)[None, :]
    m = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    m[0, :] /= np.sqrt(2.0)
    return m


def dct_matrix(n: int, dtype=np.float32) -> np.ndarray:
    return _dct_matrix64(n).astype(dtype)


def dct2(x: np.ndarray) -> np.ndarray:
    h, w = x.shape[-2:]
    mh, mw = dct_matrix(h, x.dtype), dct_matrix(w, x.dtype)
    return mh @ x @ mw.T


def idct2(c: np.ndarray) -> np.ndarray:
    h, w = c.shape[-2:]
    mh, mw = dct_matrix(h, c.dtype), dct_matrix(w, c.dtype)
    return mh.T @ c @ mw


@lru_cache(maxsize=None)
def band_map(h: int, w: int, n_bands: int) -> np.ndarray:
    i = np.arange(h)[:, None] / h
    j = np.arange(w)[None, :] / w
    f = (i + j) / 2.0
    b = np.minimum((f * n_bands).astype(np.int64), n_bands - 1)
    b[0, 0] = -1
    return b
