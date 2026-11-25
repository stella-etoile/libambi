from __future__ import annotations

import numpy as np

from .config import CodecConfig

N_AREA_BUCKETS = 9


def area_bucket(h: int, w: int) -> int:
    return int(np.clip(int(np.floor(np.log2(h * w))) - 4, 0, N_AREA_BUCKETS - 1))


def table_shapes(cfg: CodecConfig) -> dict[str, tuple]:
    nq, nb = cfg.n_q, cfg.n_bands
    shapes = {
        "ac_mag": (2, nb, nq, cfg.ac_escape + 1),
        "dc_cat": (2, nq, N_AREA_BUCKETS, cfg.dc_cats),
        "ts_mag": (2, nq, cfg.ac_escape + 1),
        "cbf": (2, 2, nq, 2),
    }
    if cfg.intra_preds:
        shapes.update({
            "ir_ac_mag": (2, nb, nq, cfg.ac_escape + 1),
            "ir_dc_cat": (2, nq, N_AREA_BUCKETS, cfg.dc_cats),
            "ir_cbf": (2, nq, 2),
            "mode": (N_AREA_BUCKETS, cfg.n_modes),
        })
    return shapes


def freqs_from_probs(p: np.ndarray, total: int) -> np.ndarray:
    n = p.shape[-1]
    p = np.maximum(p, 0)
    p = p / p.sum(axis=-1, keepdims=True)
    f = 1 + np.floor(p * (total - n)).astype(np.int64)
    rem = total - f.sum(axis=-1)
    top = np.argmax(p, axis=-1)
    np.put_along_axis(f, top[..., None], np.take_along_axis(f, top[..., None], -1) + rem[..., None], -1)
    assert (f >= 1).all() and (f.sum(axis=-1) == total).all()
    return f.astype(np.int32)


def _geometric(rho: np.ndarray, n: int) -> np.ndarray:
    m = np.arange(n - 1)
    body = rho[..., None] ** m
    tail = (rho ** (n - 1) / (1 - rho))[..., None]
    p = np.concatenate([body, tail], axis=-1)
    return p / p.sum(axis=-1, keepdims=True)


class EntropyModel:

    def __init__(self, cfg: CodecConfig, freqs: dict[str, np.ndarray] | None = None):
        self.cfg = cfg
        self.shapes = table_shapes(cfg)
        self.prior_p = self._build_prior()
        if freqs is None:
            freqs = {k: freqs_from_probs(p, cfg.total) for k, p in self.prior_p.items()}
        self.set_freqs(freqs)

    def _build_prior(self) -> dict[str, np.ndarray]:
        cfg = self.cfg
        nq, nb = cfg.n_q, cfg.n_bands
        steps = np.array(cfg.qsteps, dtype=np.float64)
        prior = {}

        f_band = (np.arange(nb) + 0.5) / nb
        mean_abs = 14.0 * np.exp(-5.0 * f_band) + 0.4            # (nb,)
        mean_abs = np.broadcast_to(mean_abs[None, :, None], (2, nb, nq)).copy()
        mean_abs[1] *= 0.6
        r = mean_abs / steps[None, None, :]
        rho = np.exp(-1.0 / np.maximum(r, 0.05))
        prior["ac_mag"] = _geometric(rho, cfg.ac_escape + 1)

        dc = np.zeros((2, nq, N_AREA_BUCKETS, cfg.dc_cats))
        cats = np.arange(cfg.dc_cats)
        for ct in range(2):
            for q in range(nq):
                for b in range(N_AREA_BUCKETS):
                    area = 2.0 ** (b + 4.5)
                    expect = (100.0 if ct == 0 else 30.0) * np.sqrt(area) / steps[q]
                    c0 = np.log2(max(expect, 1.0)) + 1
                    p = np.exp(-((cats - c0) ** 2) / (2 * 2.5**2))
                    dc[ct, q, b] = p / p.sum()
        prior["dc_cat"] = dc

        r_ts = 4.0 / steps[None, :]
        r_ts = np.broadcast_to(r_ts, (2, nq)).copy()
        rho_ts = np.exp(-1.0 / np.maximum(r_ts, 0.05))
        prior["ts_mag"] = _geometric(rho_ts, cfg.ac_escape + 1)

        prior["cbf"] = np.full((2, 2, nq, 2), 0.5)

        if cfg.intra_preds:
            mean_r = 4.0 * np.exp(-3.0 * f_band) + 0.3
            mean_r = np.broadcast_to(mean_r[None, :, None], (2, nb, nq)).copy()
            mean_r[1] *= 0.6
            rr = mean_r / steps[None, None, :]
            prior["ir_ac_mag"] = _geometric(np.exp(-1.0 / np.maximum(rr, 0.05)), cfg.ac_escape + 1)
            irdc = np.zeros((2, nq, N_AREA_BUCKETS, cfg.dc_cats))
            for ct in range(2):
                for q in range(nq):
                    for b in range(N_AREA_BUCKETS):
                        area = 2.0 ** (b + 4.5)
                        expect = (6.0 if ct == 0 else 3.0) * np.sqrt(area) / steps[q]
                        c0 = np.log2(max(expect, 1.0)) + 0.5
                        p = np.exp(-((cats - c0) ** 2) / (2 * 2.0**2))
                        irdc[ct, q, b] = p / p.sum()
            prior["ir_dc_cat"] = irdc
            prior["ir_cbf"] = np.full((2, nq, 2), 0.5)
            prior["mode"] = np.full((N_AREA_BUCKETS, cfg.n_modes), 1.0 / cfg.n_modes)
        return prior

    def set_freqs(self, freqs: dict[str, np.ndarray]) -> None:
        total = self.cfg.total
        for k, shape in self.shapes.items():
            f = np.asarray(freqs[k])
            if f.shape != shape:
                raise ValueError(f"table {k}: expected {shape}, got {f.shape}")
            if (f < 1).any() or not (f.sum(axis=-1) == total).all():
                raise ValueError(f"table {k}: freqs must be >=1 and sum to {total}")
        self.freqs = {k: np.asarray(freqs[k], dtype=np.int32) for k in self.shapes}
        self.cost = {
            k: (-np.log2(f.astype(np.float64) / total)).astype(np.float32)
            for k, f in self.freqs.items()
        }

    def cdf(self, name: str, ctx: tuple) -> np.ndarray:
        f = self.freqs[name][ctx]
        return np.concatenate([[0], np.cumsum(f)]).astype(np.int64)

    def new_counts(self) -> dict[str, np.ndarray]:
        return {k: np.zeros(s, dtype=np.int64) for k, s in self.shapes.items()}

    def refit(self, counts: dict[str, np.ndarray], smoothing: float = 8.0) -> None:
        freqs = {}
        for k in self.shapes:
            c = counts[k].astype(np.float64)
            n = c.sum(axis=-1, keepdims=True)
            p = (c + smoothing * self.prior_p[k]) / (n + smoothing)
            freqs[k] = freqs_from_probs(p, self.cfg.total)
        self.set_freqs(freqs)

    def save(self, path) -> None:
        np.savez_compressed(path, **{f"f_{k}": v for k, v in self.freqs.items()})

    @classmethod
    def load(cls, cfg: CodecConfig, path) -> "EntropyModel":
        z = np.load(path)
        return cls(cfg, {k: z[f"f_{k}"] for k in table_shapes(cfg)})


def eg0_bits(v: np.ndarray) -> np.ndarray:
    return 2 * np.floor(np.log2(v.astype(np.float64) + 1)).astype(np.float32) + 1
