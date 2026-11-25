from __future__ import annotations

import numpy as np

from .distortion import C1, C2, box_mean


def rect_steps(cfg, rects) -> list:
    out = []
    for (x, y, w, h, mode) in rects:
        q = cfg.mode_q(mode)
        step = cfg.deblock_flat_step if q < 0 else cfg.qsteps[q]
        out.append((x, y, w, h, step))
    return out


def _paint(rects_s, H: int, W: int):
    ids = np.full((H, W), -1, dtype=np.int32)
    steps = np.zeros((H, W), dtype=np.float32)
    for i, (x, y, w, h, s) in enumerate(rects_s):
        ids[y:y + h, x:x + w] = i
        steps[y:y + h, x:x + w] = s
    return ids, steps


def _filter_pass(c: np.ndarray, ids: np.ndarray, steps: np.ndarray, cfg, gain: float) -> np.ndarray:
    ys, xs = np.nonzero(ids[:, 1:] != ids[:, :-1])
    if ys.size == 0:
        return c
    x = xs + 1                                     # first pixel of the q side
    s = np.maximum(steps[ys, x - 1], steps[ys, x])           # (n,)
    scale = np.array([1.0, cfg.chroma_step_scale, cfg.chroma_step_scale], dtype=np.float32)[:, None]
    st = s[None, :] * scale                                  # (3, n)
    p2, p1, p0 = c[:, ys, x - 3], c[:, ys, x - 2], c[:, ys, x - 1]
    q0, q1, q2 = c[:, ys, x], c[:, ys, x + 1], c[:, ys, x + 2]
    a_t = cfg.deblock_alpha * gain * st
    b_t = cfg.deblock_beta * gain * st
    tc = cfg.deblock_tc * st
    ok = (np.abs(q0 - p0) < a_t) & (np.abs(p1 - p0) < b_t) & (np.abs(q1 - q0) < b_t)
    d0 = np.clip((4.0 * (q0 - p0) + (p1 - q1)) / 8.0, -tc, tc) * ok
    mid = 0.5 * (p0 + q0)
    dp1 = np.clip((p2 + mid - 2.0 * p1) / 2.0, -tc, tc) * (ok & (np.abs(p2 - p0) < b_t))
    dq1 = np.clip((q2 + mid - 2.0 * q1) / 2.0, -tc, tc) * (ok & (np.abs(q2 - q0) < b_t))
    out = c.copy()
    out[:, ys, x - 1] = p0 + d0
    out[:, ys, x] = q0 - d0
    out[:, ys, x - 2] = p1 + dp1
    out[:, ys, x + 1] = q1 + dq1
    return out


def deblock(canvas: np.ndarray, rects_s, cfg, level: int) -> np.ndarray:
    if level <= 0:
        return canvas
    gain = float(level) * cfg.deblock_gain_step
    _, H, W = canvas.shape
    ids, steps = _paint(rects_s, H, W)
    c = canvas.astype(np.float32, copy=False)
    c = _filter_pass(c, ids, steps, cfg, gain)
    ct = np.ascontiguousarray(c.transpose(0, 2, 1))
    ct = _filter_pass(ct, np.ascontiguousarray(ids.T), np.ascontiguousarray(steps.T), cfg, gain)
    return np.ascontiguousarray(ct.transpose(0, 2, 1))


def image_distortion(cfg, orig_ycc: np.ndarray, rec_ycc: np.ndarray, height: int, width: int) -> float:
    a = orig_ycc[:, :height, :width].astype(np.float64)
    b = rec_ycc[:, :height, :width].astype(np.float64)
    mse = float(np.mean((a - b) ** 2)) / 255.0**2
    k = max(1, min(cfg.ssim_win, height, width))
    if k % 2 == 0:
        k -= 1
    x, y = a[0][None] / 255.0, b[0][None] / 255.0
    mx, my = box_mean(x, k), box_mean(y, k)
    vx = box_mean(x * x, k) - mx**2
    vy = box_mean(y * y, k) - my**2
    cov = box_mean(x * y, k) - mx * my
    ssim = float((((2 * mx * my + C1) * (2 * cov + C2)) / ((mx**2 + my**2 + C1) * (vx + vy + C2))).mean())
    return cfg.alpha * cfg.mse_scale * mse + (1.0 - cfg.alpha) * (1.0 - ssim)


def choose_level(cfg, orig_ycc, rec_ycc, rects_s, height: int, width: int):
    scores = []
    for lv in range(cfg.deblock_max_level + 1):
        out = deblock(rec_ycc, rects_s, cfg, lv)
        scores.append(image_distortion(cfg, orig_ycc, out, height, width))
    best = int(np.argmin(scores))
    return best, scores
