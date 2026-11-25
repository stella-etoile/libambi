from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .bitstream import EncodeResult, decode_image, encode_image, pad_to_tiles, tile_grid
from .color import psnr_rgb, rgb_to_ycc
from .intra import new_ext
from .oracle import build_tile_table, solve_table
from .profile import Profile

MAX_TABLE_TILES = 96
SAMPLE_TILES = 12
HEADER_AND_FLUSH_BYTES = 19
RGB_K = np.array([3.0, 3.2584, 2.4756])


@dataclass
class TargetResult:
    result: EncodeResult
    lam: float
    est_bytes: float
    est_psnr: float
    real_psnr: float | None
    reached: bool
    iterations: int
    used_sample: bool


class TileTables:
    def __init__(self, rgb: np.ndarray, profile: Profile, progress=None):
        cfg = profile.cfg
        H, W = rgb.shape[:2]
        self.cfg = cfg
        self.n_px_image = H * W
        ty, tx = tile_grid(H, W, cfg.tile)
        self.n_tiles = ty * tx
        ycc = pad_to_tiles(rgb_to_ycc(rgb), cfg.tile)
        T = cfg.tile
        self.full = self.n_tiles <= max(16, MAX_TABLE_TILES * 9 // cfg.n_modes)
        if self.full:
            idxs = list(range(self.n_tiles))
        else:
            idxs = sorted(set(np.linspace(0, self.n_tiles - 1, SAMPLE_TILES).round().astype(int).tolist()))
        self.indices = idxs
        self.tables = []
        ext_orig = None
        if cfg.intra_preds:
            ext_orig = new_ext(ycc.shape[1], ycc.shape[2])
            ext_orig[:, 1:, 1:] = ycc
        for n, idx in enumerate(idxs):
            j, i = divmod(idx, tx)
            tile = np.ascontiguousarray(ycc[:, j * T:(j + 1) * T, i * T:(i + 1) * T])
            ext = None if ext_orig is None else ext_orig[:, j * T:(j + 1) * T + 1, i * T:(i + 1) * T + 1]
            self.tables.append(build_tile_table(cfg, profile.model, tile) if ext is None
                               else build_tile_table(cfg, profile.model, tile, ext=ext))
            if progress:
                progress(n + 1, len(idxs))

    def estimate(self, lam: float, mu: float = 0.0):
        bits = 0.0
        sse = np.zeros(3)
        for t in self.tables:
            r = solve_table(self.cfg, t, lam, mu)
            bits += r.total_bits
            sse += r.sse
        scale = self.n_tiles / len(self.tables)
        nbytes = bits * scale / 8 + HEADER_AND_FLUSH_BYTES
        sse_rgb = float((RGB_K * sse).sum()) * scale
        n_vals = 3.0 * self.cfg.tile**2 * self.n_tiles
        psnr = 10 * math.log10(255.0**2 * n_vals / max(sse_rgb, 1e-9))
        return nbytes, psnr


def _bisect(fn, lo, hi, want_low_lambda: bool, iters=16):
    if want_low_lambda:
        if not fn(hi):
            return hi, False
        if fn(lo):
            return lo, True
        a, b = math.log(lo), math.log(hi)
        for _ in range(iters):
            m = (a + b) / 2
            if fn(math.exp(m)):
                b = m
            else:
                a = m
        return math.exp(b), True
    if not fn(lo):
        return lo, False
    if fn(hi):
        return hi, True
    a, b = math.log(lo), math.log(hi)
    for _ in range(iters):
        m = (a + b) / 2
        if fn(math.exp(m)):
            a = m
        else:
            b = m
    return math.exp(a), True


def encode_to_target(rgb: np.ndarray, profile: Profile, *, max_bytes: int | None = None,
                     target_psnr: float | None = None, mu_bits: float = 0.0,
                     tables: TileTables | None = None, progress=None, max_iter: int = 3,
                     lam_range=(1e-4, 20.0), deblock: int | None = None) -> TargetResult:
    if (max_bytes is None) == (target_psnr is None):
        raise ValueError("give exactly one of max_bytes or target_psnr")
    tabs = tables or TileTables(rgb, profile, progress=progress)
    goal_bytes, goal_psnr = (float(max_bytes) if max_bytes else None), target_psnr
    best_ok = None
    last = None
    for it in range(1, max_iter + 1):
        if goal_bytes is not None:
            fn = lambda lam: tabs.estimate(lam, mu_bits)[0] <= goal_bytes
            lam, found = _bisect(fn, *lam_range, want_low_lambda=True)
        else:
            fn = lambda lam: tabs.estimate(lam, mu_bits)[1] >= goal_psnr
            lam, found = _bisect(fn, *lam_range, want_low_lambda=False)
        est_b, est_p = tabs.estimate(lam, mu_bits)
        res = encode_image(rgb, profile, lam, mu_bits, tables=tabs.tables if tabs.full else None,
                           deblock=deblock)
        real_p = psnr_rgb(rgb, decode_image(res.data, profile))
        ok = found and ((goal_bytes is None or res.nbytes <= max_bytes) and
                        (goal_psnr is None or real_p >= target_psnr))
        last = TargetResult(res, lam, est_b, est_p, real_p, ok, it, not tabs.full)
        if ok:
            if best_ok is None or (goal_bytes is not None and real_p > best_ok.real_psnr) or \
               (goal_psnr is not None and res.nbytes < best_ok.result.nbytes):
                best_ok = last
            break
        if not found:
            break
        if goal_bytes is not None:
            goal_bytes -= max(res.nbytes - max_bytes, 0) + 1
        else:
            goal_psnr += max(target_psnr - real_p, 0) + 0.02
    return best_ok or last


def rd_curve(tables: TileTables, lams, mu_bits: float = 0.0):
    """Estimated (lambda, bytes, psnr) points from the tables, no pixel work."""
    return [(lam, *tables.estimate(lam, mu_bits)) for lam in lams]
