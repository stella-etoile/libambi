from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CodecConfig, bits_for_choices
from .distortion import SsimContext, mse_term, ssim_term
from .entropy import EntropyModel, area_bucket
from .intra import gather_refs, new_ext, predict, predict_all
from .modes import LeafCoder


@dataclass
class Leaf:
    x: int
    y: int
    w: int
    h: int
    mode: int
    bits: float = 0.0
    dist: float = 0.0


@dataclass
class Node:
    x: int
    y: int
    w: int
    h: int
    kind: str
    k: int = 0
    mode: int = -1
    a: "Node | None" = None
    b: "Node | None" = None
    qv: object = None
    mean: object = None


@dataclass
class OracleResult:
    leaves: list
    total_cost: float
    tree_bits: float
    leaf_bits: float
    dist: float
    n_pixels: int
    tree: Node | None = None
    sse: np.ndarray | None = None

    @property
    def total_bits(self) -> float:
        return self.tree_bits + self.leaf_bits

    @property
    def bpp(self) -> float:
        return self.total_bits / self.n_pixels


def node_bits(cfg: CodecConfig, w: int, h: int) -> dict:
    legal = cfg.leaf_legal(w, h)
    can_cut = w > 1 or h > 1
    flag = 1.0 if (legal and can_cut) else 0.0
    axis = 1.0 if (w > 1 and h > 1) else 0.0
    return {
        "leaf": flag,
        "cut_v": flag + axis + bits_for_choices(w - 1),
        "cut_h": flag + axis + bits_for_choices(h - 1),
    }


def pad_luma(cfg: CodecConfig, tile: np.ndarray) -> np.ndarray:
    r = cfg.ssim_win // 2
    return np.pad(tile[0] / 255.0, r, mode="reflect").astype(np.float32)


def mode_cost(cfg: CodecConfig, model: EntropyModel, modes, w: int, h: int) -> np.ndarray:
    if cfg.intra_preds:
        bucket = area_bucket(h * cfg.grid, w * cfg.grid)
        return model.cost["mode"][bucket][np.asarray(modes, dtype=np.int64)].astype(np.float32)
    return np.full(len(modes), bits_for_choices(len(modes)), dtype=np.float32)


def tile_ext(tile: np.ndarray) -> np.ndarray:
    t = tile.shape[1]
    ext = new_ext(t, t)
    ext[:, 1:, 1:] = tile
    return ext


class LeafEvaluator:
    def __init__(self, cfg: CodecConfig, model: EntropyModel):
        self.cfg = cfg
        self.model = model
        self.coder = LeafCoder(cfg, model)

    def _code(self, m, b, cache, preds):
        if self.cfg.mode_family(m) == "intra":
            return self.coder.code_batch(m, b, cache=cache, pred=preds[self.cfg.mode_pred(m)])
        return self.coder.code_batch(m, b, cache=cache)

    def _prepare(self, tile, tile_pad_y, w, h, ext=None):
        cfg, g = self.cfg, self.cfg.grid
        r = cfg.ssim_win // 2
        H, W = h * g, w * g
        win = np.lib.stride_tricks.sliding_window_view(tile, (H, W), axis=(1, 2))[:, ::g, ::g]
        _, ny, nx, _, _ = win.shape
        b = np.ascontiguousarray(win.transpose(1, 2, 0, 3, 4).reshape(ny * nx, 3, H, W))
        pw = np.lib.stride_tricks.sliding_window_view(tile_pad_y, (H + 2 * r, W + 2 * r))[::g, ::g]
        patches = np.ascontiguousarray(pw.reshape(ny * nx, H + 2 * r, W + 2 * r))
        preds = None
        if cfg.intra_preds:
            e = tile_ext(tile) if ext is None else ext
            preds = predict_all(cfg.intra_preds, e, np.repeat(np.arange(ny) * g, nx), np.tile(np.arange(nx) * g, ny), H, W)
        return H, W, ny, nx, b, SsimContext(patches, cfg.ssim_win), preds

    def evaluate_shape(self, tile, tile_pad_y, w, h, lam, ext=None):
        cfg = self.cfg
        H, W, ny, nx, b, ssim_ctx, preds = self._prepare(tile, tile_pad_y, w, h, ext)
        modes = cfg.legal_modes(w, h)
        mbits = mode_cost(cfg, self.model, modes, w, h)
        n = ny * nx
        cache: dict = {}
        recon, lower, bits_m, sse_m = {}, {}, {}, {}
        for mi, m in enumerate(modes):
            rc, bits = self._code(m, b, cache, preds)
            sse = ((rc - b) ** 2).sum(axis=(1, 2, 3))
            bits = bits + mbits[mi]
            recon[m], bits_m[m], sse_m[m] = rc, bits, sse
            lower[m] = mse_term(cfg, sse) + lam * bits
        order = sorted(modes, key=lambda m: float(lower[m].mean()))

        best = np.full(n, np.inf)
        best_mode = np.full(n, -1, dtype=np.int32)
        best_bits = np.zeros(n)
        best_dist = np.zeros(n)
        for m in order:
            idx = np.nonzero(lower[m] < best)[0]
            if idx.size == 0:
                continue
            ssum = ssim_ctx.ssim_sums(recon[m][idx, 0], idx)
            dist = mse_term(cfg, sse_m[m][idx]) + ssim_term(cfg, H * W, ssum)
            j = dist + lam * bits_m[m][idx]
            better = j < best[idx]
            sel = idx[better]
            best[sel] = j[better]
            best_mode[sel] = m
            best_bits[sel] = bits_m[m][sel]
            best_dist[sel] = dist[better]
        shp = (ny, nx)
        return (best.reshape(shp), best_mode.reshape(shp),
                best_bits.reshape(shp), best_dist.reshape(shp))

    def table_shape(self, tile, tile_pad_y, w, h, ext=None):
        cfg = self.cfg
        H, W, ny, nx, b, ssim_ctx, preds = self._prepare(tile, tile_pad_y, w, h, ext)
        modes = cfg.legal_modes(w, h)
        mbits = mode_cost(cfg, self.model, modes, w, h)
        n, M = ny * nx, len(modes)
        bits = np.zeros((n, M), dtype=np.float32)
        dist = np.zeros((n, M), dtype=np.float32)
        sse3 = np.zeros((n, M, 3), dtype=np.float32)
        cache: dict = {}
        for mi, m in enumerate(modes):
            rc, bt = self._code(m, b, cache, preds)
            s3 = ((rc - b) ** 2).sum(axis=(2, 3))
            ssum = ssim_ctx.ssim_sums(rc[:, 0])
            bits[:, mi] = bt + mbits[mi]
            sse3[:, mi] = s3
            dist[:, mi] = mse_term(cfg, s3.sum(axis=1)) + ssim_term(cfg, H * W, ssum)
        return (np.array(modes, dtype=np.int32), bits.reshape(ny, nx, M),
                dist.reshape(ny, nx, M), sse3.reshape(ny, nx, M, 3))


@dataclass
class TileTable:
    shapes: dict


def build_tile_table(cfg: CodecConfig, model: EntropyModel, tile: np.ndarray,
                     evaluator: LeafEvaluator | None = None, ext: np.ndarray | None = None) -> TileTable:
    ev = evaluator or LeafEvaluator(cfg, model)
    pad_y = pad_luma(cfg, tile)
    n = cfg.n_cells
    shapes = {}
    for w in range(1, n + 1):
        for h in range(1, n + 1):
            if cfg.leaf_legal(w, h):
                shapes[w, h] = ev.table_shape(tile, pad_y, w, h, ext)
    return TileTable(shapes)


def _dp_values(cfg, leaf_j, lam, mu_bits):
    n = cfg.n_cells
    best, choice, leafc = {}, {}, {}
    for w in range(1, n + 1):
        for h in range(1, n + 1):
            ny, nx = n - h + 1, n - w + 1
            nb = node_bits(cfg, w, h)
            if (w, h) in leaf_j:
                cur = leaf_j[w, h] + lam * (nb["leaf"] + mu_bits)
            else:
                cur = np.full((ny, nx), np.inf)
            leafc[w, h] = cur
            ch = np.zeros((ny, nx), dtype=np.int16)
            for k in range(1, w):
                cand = best[k, h][:, 0:nx] + best[w - k, h][:, k : k + nx] + lam * nb["cut_v"]
                upd = cand < cur
                cur = np.where(upd, cand, cur)
                ch = np.where(upd, k, ch)
            for k in range(1, h):
                cand = best[w, k][0:ny, :] + best[w, h - k][k : k + ny, :] + lam * nb["cut_h"]
                upd = cand < cur
                cur = np.where(upd, cand, cur)
                ch = np.where(upd, -k, ch)
            best[w, h], choice[w, h] = cur, ch
    return best, choice, leafc


def _dp(cfg, leaf_j, leaf_mode, leaf_bits, leaf_dist, leaf_sse, lam, mu_bits) -> OracleResult:
    n = cfg.n_cells
    best, choice, _ = _dp_values(cfg, leaf_j, lam, mu_bits)

    leaves: list[Leaf] = []
    tree_bits = 0.0
    sse = np.zeros(3) if leaf_sse is not None else None
    g = cfg.grid

    def walk(x, y, w, h) -> Node:
        nonlocal tree_bits, sse
        c = int(choice[w, h][y, x])
        nb = node_bits(cfg, w, h)
        if c == 0:
            tree_bits += nb["leaf"] + mu_bits
            mode = int(leaf_mode[w, h][y, x])
            leaves.append(Leaf(x * g, y * g, w * g, h * g, mode,
                               float(leaf_bits[w, h][y, x]), float(leaf_dist[w, h][y, x])))
            if sse is not None:
                sse += leaf_sse[w, h][y, x]
            return Node(x, y, w, h, "leaf", mode=mode)
        if c > 0:
            tree_bits += nb["cut_v"]
            a = walk(x, y, c, h)
            b = walk(x + c, y, w - c, h)
            return Node(x, y, w, h, "v", c, a=a, b=b)
        k = -c
        tree_bits += nb["cut_h"]
        a = walk(x, y, w, k)
        b = walk(x, y + k, w, h - k)
        return Node(x, y, w, h, "h", k, a=a, b=b)

    root = walk(0, 0, n, n)
    total = float(best[n, n][0, 0])
    lb = sum(l.bits for l in leaves)
    dist = sum(l.dist for l in leaves)
    return OracleResult(leaves, total, tree_bits, lb, dist, cfg.tile * cfg.tile, root, sse)


def solve_tile(cfg: CodecConfig, model: EntropyModel, tile: np.ndarray, lam: float,
               mu_bits: float = 0.0, evaluator: LeafEvaluator | None = None,
               ext: np.ndarray | None = None) -> OracleResult:
    n = cfg.n_cells
    assert tile.shape == (3, cfg.tile, cfg.tile)
    ev = evaluator or LeafEvaluator(cfg, model)
    pad_y = pad_luma(cfg, tile)
    leaf_j, leaf_mode, leaf_bits, leaf_dist = {}, {}, {}, {}
    for w in range(1, n + 1):
        for h in range(1, n + 1):
            if cfg.leaf_legal(w, h):
                leaf_j[w, h], leaf_mode[w, h], leaf_bits[w, h], leaf_dist[w, h] = \
                    ev.evaluate_shape(tile, pad_y, w, h, lam, ext)
    return _dp(cfg, leaf_j, leaf_mode, leaf_bits, leaf_dist, None, lam, mu_bits)


def _table_leaf_arrays(table: TileTable, lam: float):
    leaf_j, leaf_mode, leaf_bits, leaf_dist, leaf_sse = {}, {}, {}, {}, {}
    for (w, h), (modes, bits, dist, sse3) in table.shapes.items():
        j = dist.astype(np.float64) + lam * bits.astype(np.float64)
        idx = np.argmin(j, axis=-1)
        take = lambda a: np.take_along_axis(a, idx[..., None], -1)[..., 0]
        leaf_j[w, h] = take(j)
        leaf_mode[w, h] = modes[idx]
        leaf_bits[w, h] = take(bits)
        leaf_dist[w, h] = take(dist)
        leaf_sse[w, h] = np.take_along_axis(sse3, idx[..., None, None], -2)[..., 0, :]
    return leaf_j, leaf_mode, leaf_bits, leaf_dist, leaf_sse


def solve_table(cfg: CodecConfig, table: TileTable, lam: float, mu_bits: float = 0.0) -> OracleResult:
    leaf_j, leaf_mode, leaf_bits, leaf_dist, leaf_sse = _table_leaf_arrays(table, lam)
    return _dp(cfg, leaf_j, leaf_mode, leaf_bits, leaf_dist, leaf_sse, lam, mu_bits)


def solve_table_values(cfg: CodecConfig, table: TileTable, lam: float, mu_bits: float = 0.0):
    leaf_j = _table_leaf_arrays(table, lam)[0]
    best, _, leafc = _dp_values(cfg, leaf_j, lam, mu_bits)
    return best, leafc


def _evaluate_rects(ev: LeafEvaluator, tile, pad_y, rects, lam, ext=None):
    cfg = ev.cfg
    g, r = cfg.grid, cfg.ssim_win // 2
    out = [None] * len(rects)
    groups: dict = {}
    for i, (x, y, w, h) in enumerate(rects):
        groups.setdefault((w, h), []).append(i)
    for (w, h), ids in groups.items():
        H, W = h * g, w * g
        b = np.stack([tile[:, rects[i][1] * g:rects[i][1] * g + H, rects[i][0] * g:rects[i][0] * g + W]
                      for i in ids])
        patches = np.stack([pad_y[rects[i][1] * g:rects[i][1] * g + H + 2 * r,
                                  rects[i][0] * g:rects[i][0] * g + W + 2 * r] for i in ids])
        ctx = SsimContext(patches, cfg.ssim_win)
        modes = cfg.legal_modes(w, h)
        mbits = mode_cost(cfg, ev.model, modes, w, h)
        preds = None
        if cfg.intra_preds:
            e = tile_ext(tile) if ext is None else ext
            preds = predict_all(cfg.intra_preds, e, [rects[i][1] * g for i in ids], [rects[i][0] * g for i in ids], H, W)
        best_j = np.full(len(ids), np.inf)
        best = [None] * len(ids)
        cache: dict = {}
        for mi, m in enumerate(modes):
            rc, bt = ev._code(m, b, cache, preds)
            s3 = ((rc - b) ** 2).sum(axis=(2, 3))
            ssum = ctx.ssim_sums(rc[:, 0])
            dist = mse_term(cfg, s3.sum(axis=1)) + ssim_term(cfg, H * W, ssum)
            bits = bt + mbits[mi]
            j = dist + lam * bits
            for q in np.nonzero(j < best_j)[0]:
                best_j[q] = j[q]
                best[q] = (m, float(bits[q]), float(dist[q]), s3[q].astype(np.float64))
        for q, i in enumerate(ids):
            out[i] = best[q]
    return out


def result_from_tree(cfg: CodecConfig, model: EntropyModel, tile: np.ndarray, root: Node,
                     lam: float, mu_bits: float = 0.0, evaluator: LeafEvaluator | None = None,
                     ext: np.ndarray | None = None) -> OracleResult:
    """Take a partition chosen elsewhere (e.g. by a policy), pick the best mode for every leaf
    exactly as the oracle would, and return the same OracleResult the oracle produces."""
    ev = evaluator or LeafEvaluator(cfg, model)
    pad_y = pad_luma(cfg, tile)
    leaf_nodes: list[Node] = []
    tree_bits = 0.0

    def walk(nd: Node):
        nonlocal tree_bits
        nb = node_bits(cfg, nd.w, nd.h)
        if nd.kind == "leaf":
            tree_bits += nb["leaf"] + mu_bits
            leaf_nodes.append(nd)
            return
        tree_bits += nb["cut_v"] if nd.kind == "v" else nb["cut_h"]
        walk(nd.a)
        walk(nd.b)

    walk(root)
    res = _evaluate_rects(ev, tile, pad_y, [(l.x, l.y, l.w, l.h) for l in leaf_nodes], lam, ext)
    g = cfg.grid
    leaves, sse = [], np.zeros(3)
    for nd, (m, bits, dist, s3) in zip(leaf_nodes, res):
        nd.mode = int(m)
        leaves.append(Leaf(nd.x * g, nd.y * g, nd.w * g, nd.h * g, int(m), bits, dist))
        sse += s3
    lb = sum(l.bits for l in leaves)
    dist = sum(l.dist for l in leaves)
    total = dist + lam * (lb + tree_bits)
    return OracleResult(leaves, float(total), tree_bits, lb, dist, cfg.tile * cfg.tile, root, sse)


# ------------------------------------------------- closed-loop (real neighbours) pass
def _leaf_nodes(cfg: CodecConfig, root: Node, mu_bits: float):
    """Leaves in decoding order (depth first, first child first) and the tree side-info bits."""
    leaves: list = []
    tree_bits = 0.0

    def walk(nd: Node):
        nonlocal tree_bits
        nb = node_bits(cfg, nd.w, nd.h)
        if nd.kind == "leaf":
            tree_bits += nb["leaf"] + mu_bits
            leaves.append(nd)
            return
        tree_bits += nb["cut_v"] if nd.kind == "v" else nb["cut_h"]
        walk(nd.a)
        walk(nd.b)

    walk(root)
    return leaves, tree_bits


def encode_tile_closed_loop(cfg: CodecConfig, model: EntropyModel, tile: np.ndarray, pad_y: np.ndarray,
                            ext: np.ndarray, x0: int, y0: int, root: Node, lam: float,
                            mu_bits: float = 0.0, coder: LeafCoder | None = None,
                            counts: dict | None = None) -> OracleResult:
    """Code a tile whose partition is already fixed, choosing every leaf's mode with the neighbours the
    decoder will have.

    ext: the reconstruction canvas (see intra.py), shared by all tiles of the image; this tile's
         top-left corner is at canvas pixel (x0, y0). Its reconstructed pixels are written into it.
    Fills node.mode / node.qv / node.mean on the leaves (the bitstream writer sends exactly these)
    and returns the same OracleResult the oracle produces, now with the real bits and distortion.
    Uses float64 like the decoder, so the encoder's reconstruction is the decoder's, bit for bit."""
    g, r = cfg.grid, cfg.ssim_win // 2
    coder = coder or LeafCoder(cfg, model, dtype=np.float64)
    leaf_nodes, tree_bits = _leaf_nodes(cfg, root, mu_bits)
    leaves, sse = [], np.zeros(3)
    n_p = cfg.n_intra
    for nd in leaf_nodes:
        lx, ly, W, H = nd.x * g, nd.y * g, nd.w * g, nd.h * g
        blk = tile[None, :, ly:ly + H, lx:lx + W]
        ctx = SsimContext(np.ascontiguousarray(pad_y[None, ly:ly + H + 2 * r, lx:lx + W + 2 * r]), cfg.ssim_win)
        modes = cfg.legal_modes(nd.w, nd.h)
        mbits = mode_cost(cfg, model, modes, nd.w, nd.h)
        preds = None
        stack = {}
        if n_p:
            above, left, corner = gather_refs(ext, [y0 + ly], [x0 + lx], H, W)
            preds = np.concatenate([predict(k, above, left, corner, H, W) for k in cfg.intra_preds], axis=0)
            stack = coder.code_intra_stack(blk, preds)
        cache: dict = {}
        recs, bits_l = [], []
        for m in modes:
            if cfg.mode_family(m) == "intra":
                rc, bt, _ = stack[cfg.mode_q(m)]
                p = cfg.mode_pred(m)
                recs.append(rc[p]); bits_l.append(float(bt[p]))
            else:
                rc, bt = coder.code_batch(m, blk, cache=cache)
                recs.append(rc[0]); bits_l.append(float(bt[0]))
        recs = np.stack(recs)                                               # (M, 3, H, W)
        s3 = ((recs - blk) ** 2).sum(axis=(2, 3))                           # (M, 3)
        ssum = ctx.ssim_sums(recs[:, 0], np.zeros(len(modes), dtype=np.int64))
        dist = mse_term(cfg, s3.sum(axis=1)) + ssim_term(cfg, H * W, ssum)
        bits_all = np.asarray(bits_l, dtype=np.float64) + mbits
        j = dist + lam * bits_all
        k = int(np.argmin(j))
        m = int(modes[k])
        nd.mode = m
        if cfg.mode_family(m) == "intra":
            nd.qv, nd.mean = stack[cfg.mode_q(m)][2][cfg.mode_pred(m)], None
        else:
            qv, mean = coder.quantize(m, blk, cache)
            nd.qv = None if qv is None else qv[0]
            nd.mean = None if mean is None else mean[0]
        if counts is not None:                                              # symbol statistics of the chosen coding
            if cfg.mode_family(m) == "intra":
                coder.bits_dct(nd.qv[None], cfg.mode_q(m), counts, intra=True)
            else:
                coder.code_batch(m, blk, counts=counts, cache=cache)
            if cfg.intra_preds:
                counts["mode"][area_bucket(H, W), m] += 1
        ext[:, 1 + y0 + ly:1 + y0 + ly + H, 1 + x0 + lx:1 + x0 + lx + W] = recs[k]
        leaves.append(Leaf(lx, ly, W, H, m, float(bits_all[k]), float(dist[k])))
        sse += s3[k]
    lb = sum(l.bits for l in leaves)
    dist = sum(l.dist for l in leaves)
    total = dist + lam * (lb + tree_bits)
    return OracleResult(leaves, float(total), tree_bits, lb, dist, cfg.tile * cfg.tile, root, sse)


# --------------------------------------------------- reconstruction / fitting
def reconstruct_tile(cfg: CodecConfig, model: EntropyModel, tile: np.ndarray, leaves: list,
                     counts: dict | None = None):
    """Re-code a tile with a fixed partition and fixed modes. Returns (recon (3,T,T), leaf_bits).
    With intra prediction the tile is coded on its own (its edge counts as the picture edge)."""
    if cfg.intra_preds:
        return _reconstruct_tile_intra(cfg, model, tile, leaves, counts)
    coder = LeafCoder(cfg, model)
    out = np.zeros_like(tile)
    bits = 0.0
    mode_bits_cache = {}
    for l in leaves:
        blk = tile[None, :, l.y : l.y + l.h, l.x : l.x + l.w]
        rc, b = coder.code_batch(l.mode, blk, counts=counts)
        out[:, l.y : l.y + l.h, l.x : l.x + l.w] = rc[0]
        key = (l.w, l.h)
        if key not in mode_bits_cache:
            mode_bits_cache[key] = bits_for_choices(
                len(cfg.legal_modes(l.w // cfg.grid, l.h // cfg.grid)))
        bits += float(b[0]) + mode_bits_cache[key]
    return out, bits


def _reconstruct_tile_intra(cfg, model, tile, leaves, counts):
    coder = LeafCoder(cfg, model, dtype=np.float64)
    ext = new_ext(cfg.tile, cfg.tile)
    out = np.zeros_like(tile)
    bits = 0.0
    g = cfg.grid
    for l in leaves:                              # `leaves` are in decoding order
        blk = tile[None, :, l.y:l.y + l.h, l.x:l.x + l.w]
        pred = None
        if cfg.mode_family(l.mode) == "intra":
            above, left, corner = gather_refs(ext, [l.y], [l.x], l.h, l.w)
            pred = predict(cfg.intra_preds[cfg.mode_pred(l.mode)], above, left, corner, l.h, l.w)
        rc, b = coder.code_batch(l.mode, blk, counts=counts, pred=pred)
        out[:, l.y:l.y + l.h, l.x:l.x + l.w] = rc[0]
        ext[:, 1 + l.y:1 + l.y + l.h, 1 + l.x:1 + l.x + l.w] = rc[0]
        modes = cfg.legal_modes(l.w // g, l.h // g)
        bits += float(b[0]) + float(mode_cost(cfg, model, [l.mode], l.w // g, l.h // g)[0])
        if counts is not None:
            counts["mode"][area_bucket(l.h, l.w), l.mode] += 1
    return out, bits


_FIT = {}


def _fit_init(cfg, model):
    _FIT["cfg"], _FIT["model"] = cfg, model


def _fit_chunk(args):
    """Worker: solve a chunk of tiles with the current tables and return the summed symbol counts."""
    tiles, lam, mu_bits = args
    cfg, model = _FIT["cfg"], _FIT["model"]
    counts = model.new_counts()
    bits = px = 0
    for t in tiles:
        res = solve_tile(cfg, model, t, lam, mu_bits)
        if cfg.intra_preds:       # statistics of what the encoder really codes: modes chosen with real neighbours
            res = encode_tile_closed_loop(cfg, model, t, pad_luma(cfg, t), new_ext(cfg.tile, cfg.tile),
                                          0, 0, res.tree, lam, mu_bits, counts=counts)
        else:
            reconstruct_tile(cfg, model, t, res.leaves, counts=counts)
        bits += res.total_bits
        px += res.n_pixels
    return counts, bits, px


def fit_entropy_model(cfg: CodecConfig, tiles: list, lam: float, iters: int = 2,
                      mu_bits: float = 0.0, model: EntropyModel | None = None,
                      verbose: bool = True, workers: int = 1) -> EntropyModel:
    """Alternate: solve partitions with the current model, refit tables on chosen leaves.

    workers > 1 solves the tiles in parallel processes (counts are additive, so the result is identical)."""
    model = model or EntropyModel(cfg)
    for it in range(iters):
        counts = model.new_counts()
        if workers > 1 and len(tiles) > 1:
            import multiprocessing as mp
            n_chunks = min(len(tiles), workers * 4)
            chunks = [tiles[i::n_chunks] for i in range(n_chunks)]
            tot_bits = tot_px = 0
            with mp.get_context("spawn").Pool(workers, initializer=_fit_init, initargs=(cfg, model)) as pool:
                for c, b, n in pool.imap_unordered(_fit_chunk, [(ch, lam, mu_bits) for ch in chunks]):
                    for k in counts:
                        counts[k] += c[k]
                    tot_bits += b
                    tot_px += n
        else:
            _fit_init(cfg, model)
            counts, tot_bits, tot_px = _fit_chunk((tiles, lam, mu_bits))
        if verbose:
            print(f"  fit iter {it}: bpp={tot_bits / tot_px:.3f} (estimated, before refit)", flush=True)
        model.refit(counts)
    return model
