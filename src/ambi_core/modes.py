from __future__ import annotations

import numpy as np

from .config import CodecConfig
from .dct import band_map, dct2, idct2
from .entropy import EntropyModel, area_bucket, eg0_bits

FLAT_BITS = 24.0


def bit_length(a: np.ndarray) -> np.ndarray:
    """Number of bits needed for non-negative ints (0 -> 0)."""
    a = a.astype(np.float64)
    out = np.zeros(a.shape, dtype=np.int64)
    pos = a > 0
    out[pos] = np.floor(np.log2(a[pos])).astype(np.int64) + 1
    return out


class LeafCoder:
    def __init__(self, cfg: CodecConfig, model: EntropyModel, dtype=np.float32):
        self.cfg = cfg
        self.model = model
        self.dtype = dtype

    def chan_steps(self, qi: int) -> np.ndarray:
        s = self.cfg.qsteps[qi]
        cs = self.cfg.chroma_step_scale
        return np.array([s, s * cs, s * cs], dtype=self.dtype).reshape(3, 1, 1)

    @staticmethod
    def block_means(b: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(b.mean(axis=(2, 3))), 0, 255).astype(np.int32)

    def _quant(self, x: np.ndarray, qi: int, dc_round: bool) -> np.ndarray:
        st = self.chan_steps(qi)
        a = np.abs(x) / st
        q = np.floor(a + self.cfg.rounding_offset)
        if dc_round:
            q[..., 0, 0] = np.floor(a[..., 0, 0] + 0.5)
        return (np.sign(x) * q).astype(np.int32)

    def recon_dct(self, qv: np.ndarray, qi: int) -> np.ndarray:
        coef = qv.astype(self.dtype) * self.chan_steps(qi)
        x = idct2(coef) + 128.0
        return np.clip(np.rint(x), 0, 255).astype(np.float32)

    def recon_intra(self, qv: np.ndarray, qi: int, pred: np.ndarray) -> np.ndarray:
        coef = qv.astype(self.dtype) * self.chan_steps(qi)
        x = idct2(coef) + pred.astype(self.dtype)
        return np.clip(np.rint(x), 0, 255).astype(np.float32)

    def recon_ts(self, qv: np.ndarray, mean: np.ndarray, qi: int) -> np.ndarray:
        x = mean[:, :, None, None].astype(self.dtype) + qv.astype(self.dtype) * self.chan_steps(qi)
        return np.clip(np.rint(x), 0, 255).astype(np.float32)

    def recon_flat(self, mean: np.ndarray, h: int, w: int) -> np.ndarray:
        return np.broadcast_to(
            mean[:, :, None, None].astype(np.float32), (mean.shape[0], 3, h, w)
        ).copy()

    def bits_dct(self, qv: np.ndarray, qi: int, counts: dict | None = None, intra: bool = False) -> np.ndarray:
        cfg, cost = self.cfg, self.model.cost
        k_dc, k_ac = ("ir_dc_cat", "ir_ac_mag") if intra else ("dc_cat", "ac_mag")
        n, _, h, w = qv.shape
        bm = band_map(h, w, cfg.n_bands)
        ac = bm >= 0
        b_ac = bm[ac]
        bucket = area_bucket(h, w)
        total = np.zeros(n, dtype=np.float32)
        for ch in range(3):
            ct = 0 if ch == 0 else 1
            a = np.abs(qv[:, ch])
            cat = np.minimum(bit_length(a[:, 0, 0]), cfg.dc_cats - 1)
            dc_cost = cost[k_dc][ct, qi, bucket][cat] + np.maximum(cat - 1, 0) + (cat > 0)
            a_ac = a[:, ac]
            m = np.minimum(a_ac, cfg.ac_escape)
            tbl = cost[k_ac][ct]
            c = tbl[b_ac[None, :], qi, m]
            extra = (a_ac > 0).astype(np.float32)
            esc = a_ac >= cfg.ac_escape
            if esc.any():
                extra = extra + np.where(esc, eg0_bits(np.maximum(a_ac - cfg.ac_escape, 0)), 0)
            ac_cost = (c + extra).sum(axis=1)
            cbf = (a_ac > 0).any(axis=1)
            cbf_cost = (cost["ir_cbf"][ct, qi] if intra else cost["cbf"][0, ct, qi])[cbf.astype(np.int64)]
            total += dc_cost + cbf_cost + np.where(cbf, ac_cost, 0)
            if counts is not None:
                counts[k_dc][ct, qi, bucket] += np.bincount(cat, minlength=cfg.dc_cats)
                (counts["ir_cbf"][ct, qi] if intra else counts["cbf"][0, ct, qi])[:] += np.bincount(cbf.astype(np.int64), minlength=2)
                if cbf.any():
                    idx = (b_ac[None, :] * (cfg.ac_escape + 1) + m)[cbf].ravel()
                    bc = np.bincount(idx, minlength=cfg.n_bands * (cfg.ac_escape + 1))
                    counts[k_ac][ct, :, qi, :] += bc.reshape(cfg.n_bands, cfg.ac_escape + 1)
        return total

    def bits_ts(self, qv: np.ndarray, qi: int, counts: dict | None = None) -> np.ndarray:
        cfg, cost = self.cfg, self.model.cost
        n = qv.shape[0]
        total = np.full(n, FLAT_BITS, dtype=np.float32)
        for ch in range(3):
            ct = 0 if ch == 0 else 1
            a = np.abs(qv[:, ch]).reshape(n, -1)
            m = np.minimum(a, cfg.ac_escape)
            c = cost["ts_mag"][ct, qi][m]
            extra = (a > 0).astype(np.float32)
            esc = a >= cfg.ac_escape
            if esc.any():
                extra = extra + np.where(esc, eg0_bits(np.maximum(a - cfg.ac_escape, 0)), 0)
            cbf = (a > 0).any(axis=1)
            cbf_cost = cost["cbf"][1, ct, qi][cbf.astype(np.int64)]
            total += cbf_cost + np.where(cbf, (c + extra).sum(axis=1), 0)
            if counts is not None:
                counts["cbf"][1, ct, qi] += np.bincount(cbf.astype(np.int64), minlength=2)
                if cbf.any():
                    counts["ts_mag"][ct, qi] += np.bincount(m[cbf].ravel(), minlength=cfg.ac_escape + 1)
        return total

    def quantize(self, mode: int, b: np.ndarray, cache: dict | None = None, pred: np.ndarray | None = None):
        fam, qi = self.cfg.mode_family(mode), self.cfg.mode_q(mode)
        if fam == "intra":
            if pred is None:
                raise ValueError("an intra mode needs its prediction")
            return self._quant(self._intra_coef(mode, b, pred, cache), qi, dc_round=True), None
        if fam == "dct":
            coef = None if cache is None else cache.get("coef")
            if coef is None:
                coef = dct2(b - 128.0)
                if cache is not None:
                    cache["coef"] = coef
            return self._quant(coef, qi, dc_round=True), None
        mean = self.block_means(b)
        if fam == "flat":
            return None, mean
        r = b - mean[:, :, None, None].astype(np.float32)
        return self._quant(r, qi, dc_round=False), mean

    def _intra_coef(self, mode: int, b: np.ndarray, pred: np.ndarray, cache: dict | None):
        key = ("icoef", self.cfg.mode_pred(mode))
        coef = None if cache is None else cache.get(key)
        if coef is None:
            coef = dct2(b.astype(self.dtype) - pred.astype(self.dtype))
            if cache is not None:
                cache[key] = coef
        return coef

    def code_batch(self, mode: int, b: np.ndarray, counts: dict | None = None,
                   cache: dict | None = None, pred: np.ndarray | None = None):
        fam, qi = self.cfg.mode_family(mode), self.cfg.mode_q(mode)
        n, _, h, w = b.shape
        qv, mean = self.quantize(mode, b, cache, pred)
        if fam == "intra":
            return self.recon_intra(qv, qi, pred), self.bits_dct(qv, qi, counts, intra=True)
        if fam == "dct":
            return self.recon_dct(qv, qi), self.bits_dct(qv, qi, counts)
        if fam == "flat":
            return self.recon_flat(mean, h, w), np.full(n, FLAT_BITS, dtype=np.float32)
        return self.recon_ts(qv, mean, qi), self.bits_ts(qv, qi, counts)

    def code_intra_stack(self, b: np.ndarray, preds: np.ndarray):
        coef = dct2(b.astype(self.dtype) - preds.astype(self.dtype))
        out = {}
        for qi in range(self.cfg.n_q):
            qv = self._quant(coef, qi, dc_round=True)
            out[qi] = (self.recon_intra(qv, qi, preds), self.bits_dct(qv, qi, None, intra=True), qv)
        return out

