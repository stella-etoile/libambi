from __future__ import annotations

import torch

from ambi_core.rects import RectSpace


@torch.no_grad()
def rollout(model, ycc: torch.Tensor, lams: torch.Tensor, greedy: bool = True, act_fn=None, max_steps: int = 80):
    dev = ycc.device
    B = ycc.shape[0]
    pool = None if act_fn is not None else model.trunk(ycc)
    fb = torch.arange(B, device=dev)
    fr = torch.full((B,), model.root, dtype=torch.long, device=dev)
    out_b, out_r, out_a = [], [], []
    for _ in range(max_steps):
        if fb.numel() == 0:
            break
        if act_fn is not None:
            a = act_fn(fb, fr)
        else:
            logits = model.heads(pool, fb, fr, lams[fb])
            a = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
        out_b.append(fb); out_r.append(fr); out_a.append(a)
        cut = a > 0
        if not cut.any():
            break
        cb_, cr_, ca_ = fb[cut], fr[cut], a[cut]
        fb = torch.cat([cb_, cb_])
        fr = torch.cat([model.child_a[cr_, ca_], model.child_b[cr_, ca_]])
    else:
        raise RuntimeError("rollout did not terminate (policy kept cutting)")
    return torch.cat(out_b), torch.cat(out_r), torch.cat(out_a)


def trace_cost(model, space_cost_fn, b, r, a, lam_of_b, n_tiles):
    leaf = a == 0
    contrib = torch.zeros(b.shape[0], dtype=torch.float64, device=b.device)
    contrib[leaf] = space_cost_fn(b[leaf], r[leaf]).double()
    cut = ~leaf
    contrib[cut] = lam_of_b[b[cut]].double() * model.side[r[cut], a[cut]].double()
    out = torch.zeros(n_tiles, dtype=torch.float64, device=b.device)
    out.index_add_(0, b, contrib)
    return out


def trees_from_trace(space: RectSpace, b, r, a, n_tiles):
    b, r, a = b.cpu().numpy(), r.cpu().numpy(), a.cpu().numpy()
    per = [dict() for _ in range(n_tiles)]
    for bi, ri, ai in zip(b, r, a):
        per[bi][int(ri)] = int(ai)
    return [space.build_tree(lambda rid, d=d: d[rid]) for d in per]
