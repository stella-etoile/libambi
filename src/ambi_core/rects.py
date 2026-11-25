from __future__ import annotations

import numpy as np

from .config import CodecConfig
from .oracle import Node, node_bits


class RectSpace:
    def __init__(self, cfg: CodecConfig):
        self.cfg = cfg
        n = self.n = cfg.n_cells
        self.n_actions = 1 + 2 * (n - 1)
        self.offset = np.zeros((n + 1, n + 1), dtype=np.int64)
        xs, ys, ws, hs = [], [], [], []
        off = 0
        for w in range(1, n + 1):
            for h in range(1, n + 1):
                self.offset[w, h] = off
                yy, xx = np.meshgrid(np.arange(n - h + 1), np.arange(n - w + 1), indexing="ij")
                xs.append(xx.ravel()); ys.append(yy.ravel())
                ws.append(np.full(xx.size, w)); hs.append(np.full(xx.size, h))
                off += xx.size
        self.n_rects = off
        self.x = np.concatenate(xs).astype(np.int16)
        self.y = np.concatenate(ys).astype(np.int16)
        self.w = np.concatenate(ws).astype(np.int16)
        self.h = np.concatenate(hs).astype(np.int16)
        self.root = int(self.idx(0, 0, n, n))
        self._build_actions()

    def idx(self, x, y, w, h):
        w, h = np.asarray(w), np.asarray(h)
        return self.offset[w, h] + np.asarray(y) * (self.n - w + 1) + np.asarray(x)

    def flatten(self, d: dict) -> np.ndarray:
        """{(w, h): (ny, nx) array} -> flat float32 array in rectangle order."""
        out = np.empty(self.n_rects, dtype=np.float32)
        for (w, h), arr in d.items():
            o = self.offset[w, h]
            out[o:o + arr.size] = arr.ravel()
        return out

    def _build_actions(self):
        n, A, R, cfg = self.n, self.n_actions, self.n_rects, self.cfg
        self.child_a = np.full((R, A), -1, dtype=np.int32)
        self.child_b = np.full((R, A), -1, dtype=np.int32)
        self.side = np.zeros((R, A), dtype=np.float32)
        self.legal = np.zeros((R, A), dtype=bool)
        for w in range(1, n + 1):
            for h in range(1, n + 1):
                ny, nx = n - h + 1, n - w + 1
                o = self.offset[w, h]
                rid = np.arange(o, o + ny * nx)
                yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
                yy, xx = yy.ravel(), xx.ravel()
                nb = node_bits(cfg, w, h)
                self.legal[rid, 0] = cfg.leaf_legal(w, h)
                for k in range(1, w):
                    a = k
                    self.child_a[rid, a] = self.idx(xx, yy, k, h)
                    self.child_b[rid, a] = self.idx(xx + k, yy, w - k, h)
                    self.side[rid, a] = nb["cut_v"]
                    self.legal[rid, a] = True
                for k in range(1, h):
                    a = (n - 1) + k
                    self.child_a[rid, a] = self.idx(xx, yy, w, k)
                    self.child_b[rid, a] = self.idx(xx, yy + k, w, h - k)
                    self.side[rid, a] = nb["cut_h"]
                    self.legal[rid, a] = True

    def decode_action(self, a: int):
        if a == 0:
            return "leaf", 0
        if a < self.n:
            return "v", a
        return "h", a - (self.n - 1)

    def costs(self, best: np.ndarray, leafc: np.ndarray, lam: float, rect_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(rect_ids)
        ca, cb = self.child_a[ids], self.child_b[ids]
        c = best[np.maximum(ca, 0)].astype(np.float64) + best[np.maximum(cb, 0)] + lam * self.side[ids]
        c[:, 0] = leafc[ids]
        c[~self.legal[ids]] = np.inf
        return c

    def build_tree(self, action_of) -> Node:
        def rec(x, y, w, h) -> Node:
            a = int(action_of(int(self.idx(x, y, w, h))))
            kind, k = self.decode_action(a)
            if kind == "leaf":
                return Node(x, y, w, h, "leaf")
            if kind == "v":
                return Node(x, y, w, h, "v", k, a=rec(x, y, k, h), b=rec(x + k, y, w - k, h))
            return Node(x, y, w, h, "h", k, a=rec(x, y, w, k), b=rec(x, y + k, w, h - k))
        return rec(0, 0, self.n, self.n)

    def tree_cost(self, root: Node, best_free_leafc: np.ndarray, lam: float) -> float:
        total = 0.0

        def rec(nd: Node):
            nonlocal total
            r = int(self.idx(nd.x, nd.y, nd.w, nd.h))
            if nd.kind == "leaf":
                total += float(best_free_leafc[r])
                return
            nb = node_bits(self.cfg, nd.w, nd.h)
            total += lam * (nb["cut_v"] if nd.kind == "v" else nb["cut_h"])
            rec(nd.a); rec(nd.b)
        rec(root)
        return total
