from __future__ import annotations

import numpy as np

from .config import CodecConfig
from .distortion import SsimContext
from .entropy import EntropyModel
from .modes import LeafCoder
from .oracle import pad_luma


def calibrate_mse_scale(cfg, tiles):
    model = EntropyModel(cfg); coder = LeafCoder(cfg, model)
    mse, ss, npx = 0.0, 0.0, 0
    r = cfg.ssim_win // 2
    for t in tiles:
        pad = pad_luma(cfg, t)
        for y in range(0, cfg.tile, 8):
            for x in range(0, cfg.tile, 8):
                blk = t[None, :, y:y+8, x:x+8]
                rc, _ = coder.code_batch(1, blk)
                mse += ((rc - blk) ** 2).sum() / (3 * 255.0**2)
                ctx = SsimContext(np.ascontiguousarray(pad[None, y:y+8+2*r, x:x+8+2*r]), cfg.ssim_win)
                ss += (64 - ctx.ssim_sums(rc[:, 0])[0])
                npx += 64
    return float(ss / mse)
