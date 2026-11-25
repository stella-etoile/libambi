from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ambi_core.config import CodecConfig
from ambi_core.rects import RectSpace

LOG_LAM_REF = math.log(0.05)


class ResBlock(nn.Module):
    def __init__(self, c, dil=1):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.gelu(self.c1(F.gelu(x))))


def _prefix(f: torch.Tensor) -> torch.Tensor:
    s = F.pad(f.cumsum(2).cumsum(3), (1, 0, 1, 0))
    return s.permute(0, 2, 3, 1).contiguous()


def _rect_sum(s, b, x, y, w, h):
    return s[b, y + h, x + w] - s[b, y, x + w] - s[b, y + h, x] + s[b, y, x]


def cell_stats(ycc: torch.Tensor, g: int = 4) -> torch.Tensor:
    B, _, T, _ = ycc.shape
    n = T // g
    x = ((ycc - 128.0) / 64.0).double()

    def csum(a):
        return a.reshape(B, n, g, n, g).sum((2, 4))
    y, cb, cr = x[:, 0], x[:, 1], x[:, 2]
    gx = torch.zeros_like(y); gx[:, :, :-1] = y[:, :, 1:] - y[:, :, :-1]
    gy = torch.zeros_like(y); gy[:, :-1, :] = y[:, 1:, :] - y[:, :-1, :]
    def boundary(a, axis):
        d = torch.zeros_like(a)
        if axis == 2:
            d[:, :, g::g] = a[:, :, g::g] - a[:, :, g - 1:-1:g]
        else:
            d[:, g::g, :] = a[:, g::g, :] - a[:, g - 1:-1:g, :]
        return d * d
    return torch.stack([csum(y), csum(y * y), csum(cb), csum(cb * cb), csum(cr), csum(cr * cr),
                        csum(gx * gx), csum(gy * gy),
                        csum(boundary(y, 2)), csum(boundary(cb, 2) + boundary(cr, 2)),     # across left cell edges
                        csum(boundary(y, 1)), csum(boundary(cb, 1) + boundary(cr, 1))], -1)  # across top cell edges


def _log_level(v):
    return torch.log1p(v.clamp_min(0) * 4096.0) / 5.0


class Pooled:
    def __init__(self, feat: torch.Tensor, ycc: torch.Tensor, g: int = 4):
        self.s1 = _prefix(feat)
        self.s2 = _prefix(feat * feat)
        self.g = g
        cs = cell_stats(ycc, g)
        self.hs = F.pad(cs.permute(0, 3, 1, 2).cumsum(2).cumsum(3), (1, 0, 1, 0)).permute(0, 2, 3, 1).contiguous()

    def moments(self, b, x, y, w, h):
        s = _rect_sum(self.hs, b, x, y, w, h)
        return s, (w * h).double() * (self.g * self.g)

    @staticmethod
    def hand(s, npx):
        mean = torch.stack([s[:, 0], s[:, 2], s[:, 4]], -1) / npx.unsqueeze(-1)
        var = torch.stack([s[:, 1], s[:, 3], s[:, 5]], -1) / npx.unsqueeze(-1) - mean * mean
        ge = torch.stack([s[:, 6], s[:, 7]], -1) / npx.unsqueeze(-1)
        feats = torch.cat([_log_level(var), _log_level(ge)], -1).float()
        return feats, mean

    def stats(self, b, x, y, w, h):
        area = (w * h).to(self.s1.dtype).unsqueeze(-1)
        m = _rect_sum(self.s1, b, x, y, w, h) / area
        v = _rect_sum(self.s2, b, x, y, w, h) / area - m * m
        return m, torch.sqrt(v.clamp_min(0) + 1e-6)


class PolicyNet(nn.Module):
    def __init__(self, cfg: CodecConfig, width: int = 64, hidden: int = 128):
        super().__init__()
        if cfg.grid != 4:
            raise ValueError("PolicyNet's trunk downsamples by 4: needs cfg.grid == 4")
        self.cfg_dict = cfg.to_dict()
        self.width, self.hidden = width, hidden
        sp = RectSpace(cfg)
        self.n, self.n_actions = sp.n, sp.n_actions
        for name in ("x", "y", "w", "h"):
            self.register_buffer("r_" + name, torch.as_tensor(getattr(sp, name), dtype=torch.long), persistent=False)
        self.register_buffer("child_a", torch.as_tensor(sp.child_a, dtype=torch.long), persistent=False)
        self.register_buffer("child_b", torch.as_tensor(sp.child_b, dtype=torch.long), persistent=False)
        self.register_buffer("legal", torch.as_tensor(sp.legal), persistent=False)
        self.register_buffer("side", torch.as_tensor(sp.side), persistent=False)
        self.root = sp.root

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(48, width, 3, stride=2, padding=1), nn.GELU())
        self.blocks = nn.Sequential(ResBlock(width, 1), ResBlock(width, 2), ResBlock(width, 1))
        C, H = width, hidden
        self.parent = nn.Sequential(nn.Linear(2 * C + 10, H), nn.GELU(), nn.Linear(H, H), nn.GELU())
        self.leaf_head = nn.Sequential(nn.Linear(H, H // 2), nn.GELU(), nn.Linear(H // 2, 1))
        Hc = 64
        self.child = nn.Sequential(nn.Linear(2 * C + 5, Hc), nn.GELU())
        self.cut_head = nn.Sequential(nn.Linear(H + 3 * Hc + 11, H), nn.GELU(), nn.Linear(H, 1))

    @staticmethod
    def prepare(ycc: torch.Tensor) -> torch.Tensor:
        return (ycc - 128.0) / 64.0

    def trunk(self, ycc: torch.Tensor) -> Pooled:
        return Pooled(self.blocks(self.stem(self.prepare(ycc))), ycc)

    def heads(self, pool: Pooled, qb: torch.Tensor, qr: torch.Tensor, qlam: torch.Tensor) -> torch.Tensor:
        n, A = self.n, self.n_actions
        x, y, w, h = self.r_x[qr], self.r_y[qr], self.r_w[qr], self.r_h[qr]
        lamf = ((torch.log(qlam) - LOG_LAM_REF) / 2.0).unsqueeze(-1).to(pool.s1.dtype)
        m, s = pool.stats(qb, x, y, w, h)
        sp, npx_p = pool.moments(qb, x, y, w, h)
        hand_p, mean_p = pool.hand(sp, npx_p)
        geom = torch.stack([w, h, x, y], -1).to(m.dtype) / n
        e = self.parent(torch.cat([m, s, geom, lamf, hand_p], -1))
        leaf = self.leaf_head(e)

        ca, cb = self.child_a[qr], self.child_b[qr]
        legal = self.legal[qr]
        ca0, cb0 = ca.clamp_min(0), cb.clamp_min(0)
        Q = qr.shape[0]
        qb_a = qb.unsqueeze(1).expand(Q, A)

        def stats_of(ids):
            fx, fy, fw, fh = (self.r_x[ids].reshape(-1), self.r_y[ids].reshape(-1),
                              self.r_w[ids].reshape(-1), self.r_h[ids].reshape(-1))
            fb = qb_a.reshape(-1)
            cm, cs = pool.stats(fb, fx, fy, fw, fh)
            sc, npx_c = pool.moments(fb, fx, fy, fw, fh)
            hand_c, mean_c = pool.hand(sc, npx_c)
            geo = torch.stack([fw, fh], -1).to(cm.dtype) / n
            feat = self.child(torch.cat([cm, cs, hand_c], -1)).reshape(Q, A, -1)
            return feat, geo.reshape(Q, A, 2), mean_c.reshape(Q, A, 3), npx_c.reshape(Q, A)

        pa, ga, mean_a, npx_a = stats_of(ca0)
        pb, gb, mean_b, npx_b = stats_of(cb0)
        between = (npx_a.unsqueeze(-1) * (mean_a - mean_p.unsqueeze(1)) ** 2
                   + npx_b.unsqueeze(-1) * (mean_b - mean_p.unsqueeze(1)) ** 2) / npx_p.view(Q, 1, 1)
        bv = torch.stack([_log_level(between[..., 0]), _log_level(between[..., 1] + between[..., 2])], -1).float()
        bxs, bys = self.r_x[cb0].reshape(-1), self.r_y[cb0].reshape(-1)
        v_axis = (torch.arange(A, device=qr.device) < n).view(1, A).expand(Q, A).reshape(-1)
        sw = torch.where(v_axis, torch.ones_like(bxs), w.unsqueeze(1).expand(Q, A).reshape(-1))
        sh = torch.where(v_axis, h.unsqueeze(1).expand(Q, A).reshape(-1), torch.ones_like(bys))
        ss, _ = pool.moments(qb_a.reshape(-1), bxs, bys, sw, sh)
        line_px = torch.where(v_axis, sh, sw).double() * pool.g
        e_y = torch.where(v_axis, ss[:, 8], ss[:, 10]) / line_px
        e_c = torch.where(v_axis, ss[:, 9], ss[:, 11]) / line_px
        edge = torch.stack([_log_level(e_y), _log_level(e_c)], -1).float().reshape(Q, A, 2)
        axis_v = (torch.arange(A, device=qr.device) < n).to(pa.dtype)
        k = torch.where(torch.arange(A, device=qr.device) < n, torch.arange(A, device=qr.device),
                        torch.arange(A, device=qr.device) - (n - 1)).to(pa.dtype)
        full = torch.where(axis_v.bool().unsqueeze(0), w.unsqueeze(1), h.unsqueeze(1)).to(pa.dtype)
        frac = (k.unsqueeze(0) / full.clamp_min(1)).unsqueeze(-1)
        extra = torch.cat([ga, gb, axis_v.view(1, A, 1).expand(Q, A, 1), frac,
                           lamf.unsqueeze(1).expand(Q, A, 1), bv, edge], -1)
        cand = torch.cat([e.unsqueeze(1).expand(Q, A, -1), pa, pb, (pa - pb).abs(), extra], -1)
        cut = self.cut_head(cand).squeeze(-1)
        logits = torch.cat([leaf, cut[:, 1:]], 1)
        return logits.masked_fill(~legal, float("-inf"))

    def forward(self, ycc, qb, qr, qlam):
        return self.heads(self.trunk(ycc), qb, qr, qlam)
