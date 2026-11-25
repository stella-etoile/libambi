from __future__ import annotations

import collections
import math
import time

import torch
import torch.nn.functional as F

from ambi_core.rects import RectSpace
from ambi_core.progress import LiveLine, fmt_time
from .data import OracleData
from .model import PolicyNet
from .rollout import rollout, trace_cost


class Trainer:
    def __init__(self, model: PolicyNet, space: RectSpace, data: OracleData, device="cpu", lr=2e-3, temp_bits=4.0,
                 tiles_per_step=8, lams_per_tile=2, uniform_per=48, seed=0,
                 regret_weight=1.0, source_weights=(1.0, 0.3, 1.0), norm_floor=0.25):
        self.model, self.space, self.data, self.device = model.to(device), space, data, device
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.temp, self.bt, self.lq, self.uni = temp_bits, tiles_per_step, lams_per_tile, uniform_per
        self.rw, self.src_w = regret_weight, source_weights
        self.norm_floor = (norm_floor * data.root_opt.median(0).values).float()
        self.g = torch.Generator().manual_seed(seed)
        self.shape_off = torch.as_tensor(space.offset, dtype=torch.long, device=device)
        n = space.n
        w = torch.arange(1, n + 1, device=device)
        self.shape_w, self.shape_h = torch.meshgrid(w, w, indexing="ij")
        self.shape_cnt = ((n - self.shape_h + 1) * (n - self.shape_w + 1))
        self._opt_act = None

    def action_costs(self, ti, li, rid):
        m, d = self.model, self.data
        ca, cb = m.child_a[rid], m.child_b[rid]
        t, l = ti.unsqueeze(1), li.unsqueeze(1)
        lam = d.lams[li].unsqueeze(1)
        c = d.best[t, l, ca.clamp_min(0)].double() + d.best[t, l, cb.clamp_min(0)].double() + lam * m.side[rid].double()
        c[:, 0] = d.leafc[ti, li, rid].double()
        return c.masked_fill(~m.legal[rid], float("inf"))

    def oracle_actions(self, ti, li, rid):
        return self.action_costs(ti, li, rid).argmin(-1)

    def _sample_uniform(self, k):
        n = self.space.n
        wi = torch.randint(0, n, (k,), generator=self.g).to(self.device)
        hi = torch.randint(0, n, (k,), generator=self.g).to(self.device)
        w, h = wi + 1, hi + 1
        cnt = (n - h + 1) * (n - w + 1)
        pos = torch.minimum((torch.rand(k, generator=self.g).to(self.device) * cnt).long(), cnt - 1)
        return self.shape_off[w, h] + pos

    def make_batch(self, use_policy: bool):
        d, m = self.data, self.model
        ti = torch.randint(0, d.n, (self.bt,), generator=self.g).to(self.device)
        pairs_t = ti.repeat_interleave(self.lq)
        pairs_l = torch.randint(0, d.n_lams, (pairs_t.numel(),), generator=self.g).to(self.device)
        P = pairs_t.numel()
        ycc = d.ycc[ti]
        pair_b = torch.arange(self.bt, device=self.device).repeat_interleave(self.lq)

        def over_pairs(act_fn):
            lam = d.lams[pairs_l].float()
            b, r, _ = rollout(m, ycc[pair_b], lam, act_fn=act_fn)
            return b, r

        def oracle_fn(fb, fr):
            return self.oracle_actions(pairs_t[fb], pairs_l[fb], fr)
        pb, pr = over_pairs(oracle_fn)
        bs, rs = [pb], [pr]
        ub = torch.arange(P, device=self.device).repeat_interleave(self.uni)
        bs.append(ub); rs.append(self._sample_uniform(ub.numel()))
        if use_policy:
            with torch.no_grad():
                lam = d.lams[pairs_l].float()
                qb, qr, _ = rollout(m, ycc[pair_b], lam)
            bs.append(qb); rs.append(qr)
        b = torch.cat(bs); r = torch.cat(rs)
        src = torch.cat([torch.full((x.numel(),), i, dtype=torch.long, device=self.device)
                         for i, x in zip((0, 1, 2), bs)]) if use_policy else torch.cat(
            [torch.full((x.numel(),), i, dtype=torch.long, device=self.device) for i, x in zip((0, 1), bs)])
        return ycc, pair_b, pairs_t, pairs_l, b, r, src

    def step(self, use_policy=True):
        ycc, pair_b, pairs_t, pairs_l, b, r, src = self.make_batch(use_policy)
        d, m = self.data, self.model
        lam_pair = d.lams[pairs_l]
        cost = self.action_costs(pairs_t[b], pairs_l[b], r)
        lam_q = lam_pair[b]
        regret_bits = (cost - cost.min(-1, keepdim=True).values) / lam_q.unsqueeze(1)
        target = torch.softmax((-regret_bits / self.temp).masked_fill(~torch.isfinite(regret_bits), -1e30), -1)
        pool = m.trunk(ycc)
        logits = m.heads(pool, pair_b[b], r, lam_q.float())
        logp = F.log_softmax(logits.masked_fill(~torch.isfinite(logits), -1e9), -1)
        w = torch.as_tensor(self.src_w, device=self.device, dtype=torch.float32)[src]
        ce = -(target.float() * logp).sum(-1)
        loss = (w * ce).sum() / w.sum()
        if self.rw:
            pi = logp.exp()
            reg_j = (cost - cost.min(-1, keepdim=True).values)
            reg_j = reg_j.masked_fill(~torch.isfinite(reg_j), 0.0).float()
            norm = torch.maximum(d.root_opt[pairs_t[b], pairs_l[b]].float(), self.norm_floor[pairs_l[b]])
            exp_reg = (pi * reg_j).sum(-1) / norm * 100.0
            loss = loss + self.rw * (w * exp_reg).sum() / w.sum()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        self.opt.step()
        with torch.no_grad():
            pick = logits.argmax(-1)
            reg = regret_bits.gather(1, pick.unsqueeze(1)).squeeze(1)
            acc = (reg < 1e-9).double().mean().item()
            return loss.item(), reg.mean().item(), acc

    @torch.no_grad()
    def evaluate(self, data: OracleData | None = None, batch=32):
        d = data or self.data
        m = self.model
        gaps = torch.zeros(d.n, d.n_lams, dtype=torch.float64)
        for li in range(d.n_lams):
            for s in range(0, d.n, batch):
                ids = torch.arange(s, min(s + batch, d.n), device=self.device)
                lam = d.lams[li].float().expand(ids.numel())
                b, r, a = rollout(m, d.ycc[ids], lam)
                leaf_fn = lambda bb, rr, ids=ids, li=li: d.leafc[ids[bb], li, rr]
                lam_of_b = d.lams[li].expand(ids.numel())
                j = trace_cost(m, leaf_fn, b, r, a, lam_of_b, ids.numel())
                gaps[s:s + ids.numel(), li] = ((j - d.root_opt[ids, li]) / d.root_opt[ids, li]).cpu() * 100
        out = {}
        for li in range(d.n_lams):
            g = gaps[:, li]
            opt = d.root_opt[:, li].cpu()
            agg = ((g / 100 * opt).sum() / opt.sum() * 100).item()
            out[float(d.lams[li])] = (g.mean().item(), g.median().item(), g.max().item(), (g < 1e-6).double().mean().item(), agg)
        return out, gaps


def fit(trainer: Trainer, steps: int, log_every=25, eval_every=0, val: OracleData | None = None, warmup_steps=0,
        lr_final_frac=0.1, log=print, on_eval=None, live: LiveLine | None = None):
    live = LiveLine() if live is None else live
    opt = trainer.opt
    base = opt.param_groups[0]["lr"]
    t0 = time.time()
    hist = []
    recent = collections.deque(maxlen=50)
    eval_s = []
    n_evals = (steps // eval_every) if (eval_every and val is not None) else 0
    last_val = None
    for it in range(1, steps + 1):
        frac = it / steps
        for g in opt.param_groups:
            g["lr"] = base * (lr_final_frac + (1 - lr_final_frac) * 0.5 * (1 + math.cos(math.pi * frac)))
        ts = time.time()
        loss, reg, acc = trainer.step(use_policy=it > warmup_steps)
        recent.append(time.time() - ts)
        hist.append((it, loss, reg, acc))
        if live.enabled:
            per = sum(recent) / len(recent)
            evals_left = max(0, n_evals - it // eval_every) if n_evals else 0
            eta = (steps - it) * per + evals_left * (sum(eval_s) / len(eval_s) if eval_s else 15.0)
            live.update(f"step {it}/{steps} {100 * it // steps}%  loss {loss:.3f}  regret {reg:.1f}b  exact {acc:.2f}  "
                        f"{per:.2f}s/step  ETA {fmt_time(eta)}" + (f"  val {last_val:.2f}%" if last_val is not None else ""))
        elif it % log_every == 0 or it == steps:
            log(f"step {it}/{steps}  loss {loss:.4f}  regret {reg:.2f} bits  exact {acc:.2f}  ({time.time() - t0:.0f}s)")
        if eval_every and val is not None and (it % eval_every == 0 or it == steps):
            te = time.time()
            live.update(f"step {it}/{steps}  validating ...")
            ev, _ = trainer.evaluate(val)
            eval_s.append(time.time() - te)
            agg = sum(v[4] for v in ev.values()) / len(ev)
            med = sum(v[1] for v in ev.values()) / len(ev)
            mean = sum(v[0] for v in ev.values()) / len(ev)
            last_val = agg
            live.clear()
            log(f"step {it}/{steps}  val: aggregate J gap {agg:.2f}% (median tile {med:.1f}%, mean of per-tile % {mean:.1f}%)   " +
                "  ".join(f"{l:g}:{v[4]:.1f}%" for l, v in ev.items()))
            if on_eval:
                on_eval(it, ev)
    live.done()
    return hist
