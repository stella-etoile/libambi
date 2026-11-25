from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np
import torch

from ambi_core.config import CodecConfig
from ambi_core.pruned_dp import CandidateSolver, init_worker, solve_task
from ambi_core.rects import RectSpace
from .model import PolicyNet
from .partitioner import load_policy


@torch.no_grad()
def expand(model: PolicyNet, pool, n_tiles: int, lam: float, k: int, chunk: int = 2048):
    dev, R = model.r_x.device, model.r_x.shape[0]
    k = min(k, model.n_actions)
    tk = torch.full((n_tiles, R, k), -1, dtype=torch.int16, device=dev)
    seen = torch.zeros(n_tiles, R, dtype=torch.bool, device=dev)
    fb = torch.arange(n_tiles, device=dev)
    fr = torch.full((n_tiles,), model.root, dtype=torch.long, device=dev)
    seen[fb, fr] = True
    while fb.numel():
        nb, nr = [], []
        for s in range(0, fb.numel(), chunk):
            b, r = fb[s:s + chunk], fr[s:s + chunk]
            logits = model.heads(pool, b, r, torch.full((b.numel(),), float(lam), device=dev))
            vals, idx = logits.topk(k, dim=-1)
            ok = torch.isfinite(vals)
            tk[b, r] = torch.where(ok, idx, torch.full_like(idx, -1)).to(torch.int16)
            cut = ok & (idx > 0)
            bb = b.unsqueeze(1).expand_as(idx)[cut]
            rr = r.unsqueeze(1).expand_as(idx)[cut]
            aa = idx[cut]
            nb += [bb, bb]
            nr += [model.child_a[rr, aa], model.child_b[rr, aa]]
        if not nb:
            break
        nb, nr = torch.cat(nb), torch.cat(nr)
        key = torch.unique(nb * R + nr)
        nb, nr = key // R, key % R
        fresh = ~seen[nb, nr]
        fb, fr = nb[fresh], nr[fresh]
        seen[fb, fr] = True
    return seen, tk


class PrunedPartitioner:
    def __init__(self, model: PolicyNet, profile, k: int = 3, device="cpu", batch: int = 8, chunk: int = 2048, workers: int = 0):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.model = model.to(device).eval()
        self.profile = profile
        self.cfg: CodecConfig = profile.cfg
        self.space = RectSpace(self.cfg)
        self.k, self.device, self.batch, self.chunk, self.workers = k, device, batch, chunk, workers
        self.progress = None
        self._solver = CandidateSolver(self.cfg, profile.model) if workers <= 1 else None
        self._pool = None
        self.reset_stats()

    @classmethod
    def load(cls, path, profile, k: int = 3, device="cpu", batch: int = 8, workers: int = 0):
        return cls(load_policy(path, device), profile, k, device, batch, workers=workers)

    def _get_pool(self):
        if self._pool is None:
            keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            old = {k: os.environ.get(k) for k in keys}
            for k in keys:
                os.environ.setdefault(k, "1")
            try:
                self._pool = mp.get_context("spawn").Pool(self.workers, init_worker, (self.cfg, self.profile.model))
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        return self._pool

    def close(self):
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.terminate()
            pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def reset_stats(self):
        self.stats = dict(tiles=0, looked_at=0, priced_as_leaf=0, seconds_policy=0.0, seconds_leaf=0.0, seconds_dp=0.0, seconds_wall=0.0)

    def summary(self) -> str:
        s, n = self.stats, max(1, self.stats["tiles"])
        R = self.space.n_rects
        txt = (f"pruned search k={self.k}: looked at {100 * s['looked_at'] / n / R:.1f}% of the {R} rectangles per tile, priced "
               f"{100 * s['priced_as_leaf'] / n / R:.1f}% as leaves; per tile {1000 * s['seconds_policy'] / n:.0f} ms policy + "
               f"{1000 * s['seconds_leaf'] / n:.0f} ms leaf pricing + {1000 * s['seconds_dp'] / n:.0f} ms DP (CPU time)")
        if self.workers > 1:
            txt += f"; {self.workers} workers: {1000 * s['seconds_wall'] / n:.0f} ms wall time per tile"
        return txt

    def _tally(self, ids, res):
        tree, n_priced, t_leaf, t_dp = res
        self.stats["tiles"] += 1
        self.stats["looked_at"] += len(ids)
        self.stats["priced_as_leaf"] += n_priced
        self.stats["seconds_leaf"] += t_leaf
        self.stats["seconds_dp"] += t_dp
        return tree

    @torch.no_grad()
    def trees(self, tiles: np.ndarray, lam: float):
        t_start = time.time()
        n = len(tiles)
        jobs = []
        for s in range(0, n, self.batch):
            chunk = np.ascontiguousarray(tiles[s:s + self.batch])
            t0 = time.time()
            x = torch.as_tensor(chunk, dtype=torch.float32, device=self.device)
            pool = self.model.trunk(x)
            seen, tk = expand(self.model, pool, len(chunk), lam, self.k, self.chunk)
            seen, tk = seen.cpu().numpy(), tk.cpu().numpy()
            self.stats["seconds_policy"] += time.time() - t0
            for i in range(len(chunk)):
                ids = np.nonzero(seen[i])[0]
                rows = tk[i][ids]
                if self.workers > 1:
                    jobs.append((ids, self._get_pool().apply_async(solve_task, ((chunk[i], ids.astype(np.int32), rows, lam),))))
                else:
                    jobs.append((ids, self._solver.solve(chunk[i], ids, rows, lam)))
                    if self.progress:
                        self.progress(len(jobs), n)
        out = []
        for j, (ids, res) in enumerate(jobs):
            out.append(self._tally(ids, res.get() if self.workers > 1 else res))
            if self.workers > 1 and self.progress:
                self.progress(j + 1, n)
        self.stats["seconds_wall"] += time.time() - t_start
        return out


def make_partitioner(path, profile, device="cpu", prune_k: int = 0, workers: int = 0):
    from .partitioner import PolicyPartitioner
    if prune_k:
        return PrunedPartitioner.load(path, profile, prune_k, device, workers=workers)
    return PolicyPartitioner.load(path, device)
