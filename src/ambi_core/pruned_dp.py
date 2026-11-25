from __future__ import annotations

import time

import numpy as np

from .config import CodecConfig
from .entropy import EntropyModel
from .oracle import LeafEvaluator, _evaluate_rects, node_bits, pad_luma
from .rects import RectSpace


class CandidateSolver:
    def __init__(self, cfg: CodecConfig, model: EntropyModel):
        self.cfg = cfg
        self.space = RectSpace(cfg)
        self.ev = LeafEvaluator(cfg, model)
        self._leaf_bits = {(w, h): node_bits(cfg, w, h)["leaf"] for w in range(1, cfg.n_cells + 1) for h in range(1, cfg.n_cells + 1)}

    def solve(self, tile: np.ndarray, ids: np.ndarray, tk: np.ndarray, lam: float):
        sp = self.space
        t0 = time.time()
        ids = np.asarray(ids)
        need = ids[(tk == 0).any(axis=1)]
        rects = [(int(sp.x[r]), int(sp.y[r]), int(sp.w[r]), int(sp.h[r])) for r in need]
        priced = _evaluate_rects(self.ev, tile, pad_luma(self.cfg, tile), rects, lam) if rects else []
        leaf_j = {int(r): p[2] + lam * (p[1] + self._leaf_bits[w, h])
                  for r, p, (_, _, w, h) in zip(need, priced, rects)}
        t1 = time.time()
        V: dict = {}
        choice: dict = {}
        child_a, child_b, side = sp.child_a, sp.child_b, sp.side
        for r, row in zip(ids.tolist(), tk.tolist()):
            best, pick = np.inf, -1
            for a in row:
                if a < 0:
                    continue
                c = leaf_j[r] if a == 0 else V[int(child_a[r, a])] + V[int(child_b[r, a])] + lam * float(side[r, a])
                if c < best:
                    best, pick = c, a
            if pick < 0:
                raise RuntimeError(f"rectangle {r} has no candidate action with a finite cost")
            V[r], choice[r] = best, pick
        tree = sp.build_tree(lambda rid: choice[rid])
        return tree, len(need), t1 - t0, time.time() - t1


_SOLVER: CandidateSolver | None = None


def init_worker(cfg: CodecConfig, model: EntropyModel) -> None:
    global _SOLVER
    _SOLVER = CandidateSolver(cfg, model)


def solve_task(args):
    return _SOLVER.solve(*args)
