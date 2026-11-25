from __future__ import annotations

import numpy as np

C1 = 0.01**2
C2 = 0.03**2


def box_mean(a: np.ndarray, k: int) -> np.ndarray:
    n, hp, wp = a.shape
    s = a[:, :, : wp - k + 1].copy()
    for d in range(1, k):
        s += a[:, :, d : wp - k + 1 + d]
    t = s[:, : hp - k + 1, :].copy()
    for d in range(1, k):
        t += s[:, d : hp - k + 1 + d, :]
    return t / float(k * k)


class SsimContext:
    def __init__(self, patches: np.ndarray, k: int):
        self.k = k
        self.r = k // 2
        self.patches = patches
        self.mu_x = box_mean(patches, k)
        self.var_x = box_mean(patches * patches, k) - self.mu_x**2

    def ssim_sums(self, recon_y: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        r, k = self.r, self.k
        m, h, w = recon_y.shape
        if idx is None:
            idx = np.arange(m)
        y = self.patches[idx].copy()
        y[:, r : r + h, r : r + w] = recon_y / 255.0
        mu_y = box_mean(y, k)
        var_y = box_mean(y * y, k) - mu_y**2
        cov = box_mean(self.patches[idx] * y, k) - self.mu_x[idx] * mu_y
        mu_x, var_x = self.mu_x[idx], self.var_x[idx]
        num = (2 * mu_x * mu_y + C1) * (2 * cov + C2)
        den = (mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2)
        return (num / den).sum(axis=(1, 2))


def mse_term(cfg, sse: np.ndarray) -> np.ndarray:
    return cfg.alpha * cfg.mse_scale * sse / (3.0 * 255.0**2)


def ssim_term(cfg, n_pix: int, ssim_sum: np.ndarray) -> np.ndarray:
    return (1.0 - cfg.alpha) * (n_pix - ssim_sum)
