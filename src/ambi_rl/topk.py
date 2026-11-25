from __future__ import annotations

import time

import numpy as np
import torch

from ambi_core.progress import LiveLine, fmt_time
from .data import OracleData
from .model import PolicyNet
from .rollout import rollout
from .train import Trainer


def shape_groups(space):
    n = space.n
    return [(int(space.offset[w, h]), (n - h + 1) * (n - w + 1)) for w in range(1, n + 1) for h in range(1, n + 1)]


@torch.no_grad()
def topk_actions(model: PolicyNet, pool, n_tiles: int, lam: float, kmax: int, chunk: int = 2048) -> torch.Tensor:
    R, dev = model.r_x.shape[0], model.r_x.device
    kmax = min(kmax, model.n_actions)
    out = torch.empty(n_tiles, R, kmax, dtype=torch.int16, device=dev)
    flat = out.view(n_tiles * R, kmax)
    for s in range(0, n_tiles * R, chunk):
        q = torch.arange(s, min(s + chunk, n_tiles * R), device=dev)
        logits = model.heads(pool, q // R, q % R, torch.full((q.numel(),), float(lam), device=dev))
        flat[s:s + q.numel()] = logits.topk(kmax, dim=-1).indices.to(torch.int16)
    return out


def _gather(table, sl, a):
    return table[sl].unsqueeze(0).expand(a.shape[0], -1, -1).gather(2, a)


@torch.no_grad()
def pruned_values(model: PolicyNet, groups, topk, leafc, lam: float, k: int) -> torch.Tensor:
    B, R, _ = topk.shape
    V = torch.zeros(B, R, dtype=torch.float64, device=topk.device)
    side = model.side.double()
    for o, cnt in groups:
        sl = slice(o, o + cnt)
        a = topk[:, sl, :k].long()
        ca, cb = _gather(model.child_a, sl, a).clamp_min(0), _gather(model.child_b, sl, a).clamp_min(0)
        va = V.gather(1, ca.reshape(B, -1)).reshape(B, cnt, -1)
        vb = V.gather(1, cb.reshape(B, -1)).reshape(B, cnt, -1)
        c = va + vb + lam * _gather(side, sl, a)
        c = torch.where(a == 0, leafc[:, sl].double().unsqueeze(-1).expand_as(c), c)
        c = c.masked_fill(~_gather(model.legal, sl, a), float("inf"))
        V[:, sl] = c.min(-1).values
    return V


@torch.no_grad()
def reached_rectangles(model: PolicyNet, groups, topk, k: int) -> torch.Tensor:
    B, R, _ = topk.shape
    dev = topk.device
    reach = torch.zeros(B, R + 1, dtype=torch.int8, device=dev)
    reach[:, model.root] = 1
    for o, cnt in reversed(groups):
        sl = slice(o, o + cnt)
        a = topk[:, sl, :k].long()
        cut = _gather(model.legal, sl, a) & (a > 0) & (reach[:, sl] > 0).unsqueeze(-1)
        for child in (model.child_a, model.child_b):
            ids = torch.where(cut, _gather(child, sl, a), torch.full_like(a, R))
            reach.scatter_(1, ids.reshape(B, -1), torch.ones(B, ids[0].numel(), dtype=torch.int8, device=dev))
    return reach[:, :R].sum(1)


@torch.no_grad()
def oracle_path_hits(tr: Trainer, ycc, tile_ids, li: int, topk, ks, tol: float = 1e-7):
    m, d = tr.model, tr.data
    lam = d.lams[li].float().expand(len(tile_ids))
    b, r, _ = rollout(m, ycc, lam, act_fn=lambda fb, fr: tr.oracle_actions(tile_ids[fb], torch.full_like(fb, li), fr))
    c = tr.action_costs(tile_ids[b], torch.full_like(b, li), r)                       # (Q, A)
    best = c.min(-1).values
    ck = c.gather(1, topk[b, r].long())                                               # (Q, kmax)
    hits = []
    for k in ks:
        got = ck[:, :min(k, ck.shape[1])].min(-1).values
        hits.append(int((got <= best + tol * best.abs() + 1e-12).sum()))
    return hits, int(b.numel())


@torch.no_grad()
def diagnose(model: PolicyNet, space, data: OracleData, ks, device="cpu", batch: int = 8, chunk: int = 2048,
             lam_idx=None, log=print, live: LiveLine | None = None):
    model = model.to(device).eval()
    ks = sorted(set(min(k, model.n_actions) for k in ks))
    kmax = max(ks)
    tr = Trainer(model, space, data, device)
    groups = shape_groups(space)
    R = space.n_rects
    lam_idx = list(range(data.n_lams)) if lam_idx is None else list(lam_idx)
    live = LiveLine() if live is None else live
    acc = {li: {k: dict(j=[], vis=[], hit=0, tot=0) for k in ks} for li in lam_idx}
    t0, total = time.time(), data.n * len(lam_idx)
    done = 0
    for s in range(0, data.n, batch):
        ids = torch.arange(s, min(s + batch, data.n), device=device)
        ycc = data.ycc[ids]
        pool = model.trunk(ycc)
        for li in lam_idx:
            lam = float(data.lams[li])
            topk = topk_actions(model, pool, len(ids), lam, kmax, chunk)
            leafc = data.leafc[ids.unsqueeze(1), li, torch.arange(R, device=device).unsqueeze(0)]
            hits, tot = oracle_path_hits(tr, ycc, ids, li, topk, ks)
            for k, h in zip(ks, hits):
                V = pruned_values(model, groups, topk, leafc, lam, k)
                a = acc[li][k]
                a["j"].append(V[:, space.root].cpu())
                a["vis"].append(reached_rectangles(model, groups, topk, k).double().cpu() / R)
                a["hit"] += h; a["tot"] += tot
            done += len(ids)
            el = time.time() - t0
            live.update(f"tiles x lambdas {done}/{total}  {fmt_time(el)}  ETA {fmt_time(el / done * (total - done))}")
    live.done()
    out = {}
    for li in lam_idx:
        opt = data.root_opt[:, li].cpu().double()
        out[float(data.lams[li])] = {}
        for k in ks:
            a = acc[li][k]
            j = torch.cat(a["j"])
            gap = (j - opt) / opt * 100
            out[float(data.lams[li])][k] = dict(
                gap_pct=float((j - opt).sum() / opt.sum() * 100), median_gap_pct=float(gap.median()),
                frac_optimal=float((gap < 1e-6).double().mean()), visited_frac=float(torch.cat(a["vis"]).mean()),
                path_recall=a["hit"] / max(1, a["tot"]))
    return out
