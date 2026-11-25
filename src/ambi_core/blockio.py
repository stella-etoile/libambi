from __future__ import annotations

import numpy as np

from .config import CodecConfig
from .errors import AmbiFormatError
from .dct import band_map
from .entropy import EntropyModel, area_bucket
from .modes import bit_length
from .rangecoder import RangeDecoder, RangeEncoder


def _enc_eg0(enc: RangeEncoder, v: int) -> None:
    nb = int(np.floor(np.log2(v + 1)))
    for _ in range(nb):
        enc.encode_raw(0, 1)
    enc.encode_raw(1, 1)
    if nb:
        enc.encode_raw(v + 1 - (1 << nb), nb)


def _dec_eg0(dec: RangeDecoder) -> int:
    nb = 0
    while dec.decode_raw(1) == 0:
        nb += 1
        if nb > 24:
            raise AmbiFormatError("corrupt data (escape code too long)")
    rest = dec.decode_raw(nb) if nb else 0
    return ((1 << nb) | rest) - 1


def encode_block(enc: RangeEncoder, cfg: CodecConfig, model: EntropyModel, mode: int,
                 h: int, w: int, qv, mean) -> None:
    fam, qi = cfg.mode_family(mode), cfg.mode_q(mode)
    tb = cfg.prob_bits
    if fam == "flat":
        for c in range(3):
            enc.encode_raw(int(mean[c]), 8)
        return
    if fam == "ts":
        for c in range(3):
            enc.encode_raw(int(mean[c]), 8)
    dctlike = fam in ("dct", "intra")
    k_dc, k_ac = ("ir_dc_cat", "ir_ac_mag") if fam == "intra" else ("dc_cat", "ac_mag")
    bm = band_map(h, w, cfg.n_bands)
    bucket = area_bucket(h, w)
    for ch in range(3):
        ct = 0 if ch == 0 else 1
        v = qv[ch]
        if dctlike:
            dc = int(v[0, 0])
            a = abs(dc)
            cat = min(int(bit_length(np.array(a))), cfg.dc_cats - 1)
            enc.encode_symbol(model.cdf(k_dc, (ct, qi, bucket)), cat, tb)
            if cat > 1:
                enc.encode_raw(a - (1 << (cat - 1)), cat - 1)
            if cat > 0:
                enc.encode_raw(1 if dc < 0 else 0, 1)
            cbf = any(int(v[i, j]) != 0 for i in range(h) for j in range(w) if (i, j) != (0, 0))
            enc.encode_symbol(model.cdf("ir_cbf", (ct, qi)) if fam == "intra" else model.cdf("cbf", (0, ct, qi)),
                              int(cbf), tb)
        else:
            cbf = bool(np.any(v != 0))
            enc.encode_symbol(model.cdf("cbf", (1, ct, qi)), int(cbf), tb)
        if not cbf:
            continue
        for i in range(h):
            for j in range(w):
                if dctlike and (i, j) == (0, 0):
                    continue
                x = int(v[i, j])
                a = abs(x)
                m = min(a, cfg.ac_escape)
                ctx = (ct, int(bm[i, j]), qi) if dctlike else (ct, qi)
                enc.encode_symbol(model.cdf(k_ac if dctlike else "ts_mag", ctx), m, tb)
                if a > 0:
                    enc.encode_raw(1 if x < 0 else 0, 1)
                if m == cfg.ac_escape:
                    _enc_eg0(enc, a - cfg.ac_escape)


def decode_block(dec: RangeDecoder, cfg: CodecConfig, model: EntropyModel, mode: int,
                 h: int, w: int):
    fam, qi = cfg.mode_family(mode), cfg.mode_q(mode)
    tb = cfg.prob_bits
    mean = None
    if fam in ("flat", "ts"):
        mean = np.array([dec.decode_raw(8) for _ in range(3)], dtype=np.int32)
    if fam == "flat":
        return None, mean
    dctlike = fam in ("dct", "intra")
    k_dc, k_ac = ("ir_dc_cat", "ir_ac_mag") if fam == "intra" else ("dc_cat", "ac_mag")
    bm = band_map(h, w, cfg.n_bands)
    bucket = area_bucket(h, w)
    qv = np.zeros((3, h, w), dtype=np.int32)
    for ch in range(3):
        ct = 0 if ch == 0 else 1
        if dctlike:
            cat = dec.decode_symbol(model.cdf(k_dc, (ct, qi, bucket)), tb)
            a = 0
            if cat > 0:
                a = (1 << (cat - 1)) + (dec.decode_raw(cat - 1) if cat > 1 else 0)
                if dec.decode_raw(1):
                    a = -a
            qv[ch, 0, 0] = a
            cbf = dec.decode_symbol(model.cdf("ir_cbf", (ct, qi)) if fam == "intra" else model.cdf("cbf", (0, ct, qi)), tb)
        else:
            cbf = dec.decode_symbol(model.cdf("cbf", (1, ct, qi)), tb)
        if not cbf:
            continue
        for i in range(h):
            for j in range(w):
                if dctlike and (i, j) == (0, 0):
                    continue
                ctx = (ct, int(bm[i, j]), qi) if dctlike else (ct, qi)
                m = dec.decode_symbol(model.cdf(k_ac if dctlike else "ts_mag", ctx), tb)
                a = m
                neg = 0
                if m > 0:
                    neg = dec.decode_raw(1)
                if m == cfg.ac_escape:
                    a = cfg.ac_escape + _dec_eg0(dec)
                qv[ch, i, j] = -a if neg else a
    return qv, mean
