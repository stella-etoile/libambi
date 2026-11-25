from __future__ import annotations

import numpy as np

from ambi_core.rects import RectSpace

SHAPES = [(tw, th) for tw in (2, 4, 8, 16) for th in (2, 4, 8, 16)]


def uniform_action_fn(space: RectSpace, tw: int, th: int):
    def act(rid: int) -> int:
        w, h = int(space.w[rid]), int(space.h[rid])
        if w > tw:
            return tw
        if h > th:
            return (space.n - 1) + th 
        return 0
    return act


def best_uniform_gap(space: RectSpace, data) -> np.ndarray:
    lams = data.lams.cpu().numpy()
    leafc, opt = data.leafc.cpu().numpy(), data.root_opt.cpu().numpy()
    out = np.zeros((data.n, data.n_lams))
    trees = {}
    for tw, th in SHAPES:
        if not space.cfg.leaf_legal(tw, th):
            continue
        trees[tw, th] = space.build_tree(uniform_action_fn(space, tw, th))
    for i in range(data.n):
        for j in range(data.n_lams):
            best = min(space.tree_cost(t, leafc[i, j], lams[j]) for t in trees.values())
            out[i, j] = (best - opt[i, j]) / opt[i, j] * 100
    return out
