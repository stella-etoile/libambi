from __future__ import annotations
import numpy as np

def _odd_leq(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return n if (n % 2 == 1) else (n - 1)

def _safe_win_size(req: int, H: int, W: int) -> int:
    m = max(1, min(H, W))
    k = _odd_leq(min(req, m))
    if k < 3 and m >= 3:
        k = 3
    return k

def _gaussian_kernel1d(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    size = int(size)
    if size < 1:
        size = 1
    if size % 2 == 0:
        size -= 1
    r = (size - 1) // 2
    x = np.arange(-r, r + 1, dtype=np.float64)
    s = sigma if size > 1 else 1.0
    k = np.exp(-(x * x) / (2.0 * s * s))
    k /= max(k.sum(), 1e-12)
    return k.astype(np.float32)

def _conv2_sep(img: np.ndarray, k1d: np.ndarray) -> np.ndarray:
    H, W, C = img.shape
    if H == 0 or W == 0:
        return np.zeros_like(img, dtype=np.float32)

    K = int(k1d.shape[0])
    if K <= 1:
        return img.astype(np.float32)
    r = (K - 1) // 2

    pad_h = np.pad(img, ((r, r), (0, 0), (0, 0)), mode="edge")
    tmp = np.zeros_like(img, dtype=np.float32)
    for i in range(K):
        tmp += k1d[i] * pad_h[i:i+H, :, :]

    pad_w = np.pad(tmp, ((0, 0), (r, r), (0, 0)), mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    for j in range(K):
        out += k1d[j] * pad_w[:, j:j+W, :]

    return out

def _to_eval_space(img: np.ndarray, on: str = "y") -> np.ndarray:
    on = (on or "y").lower()
    if on.startswith("y"):
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        y = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
        return y[..., None]
    return img.astype(np.float32)

def _down2(img: np.ndarray) -> np.ndarray:
    H, W, C = img.shape
    if H < 2 or W < 2:
        return img
    H2 = (H // 2) * 2
    W2 = (W // 2) * 2
    x = img[:H2, :W2, :]
    out = 0.25 * (x[0::2, 0::2, :] + x[1::2, 0::2, :] + x[0::2, 1::2, :] + x[1::2, 1::2, :])
    return out.astype(np.float32)

def _ssim_components(x: np.ndarray, y: np.ndarray, k1: float, k2: float, win: np.ndarray):
    c1 = (k1 ** 2)
    c2 = (k2 ** 2)

    mu_x = _conv2_sep(x, win)
    mu_y = _conv2_sep(y, win)

    x2 = x * x
    y2 = y * y
    xy = x * y

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = _conv2_sep(x2, win) - mu_x2
    sigma_y2 = _conv2_sep(y2, win) - mu_y2
    sigma_xy = _conv2_sep(xy, win) - mu_xy

    eps = 1e-12
    sigma_x = np.sqrt(np.maximum(sigma_x2, 0.0))
    sigma_y = np.sqrt(np.maximum(sigma_y2, 0.0))

    l = (2.0 * mu_xy + c1) / (mu_x2 + mu_y2 + c1 + eps)
    c = (2.0 * sigma_x * sigma_y + c2) / (sigma_x2 + sigma_y2 + c2 + eps)
    s = (sigma_xy + c2 / 2.0) / (sigma_x * sigma_y + c2 / 2.0 + eps)

    l_m = l.mean(axis=(0, 1))
    c_m = c.mean(axis=(0, 1))
    s_m = s.mean(axis=(0, 1))
    return l_m, c_m, s_m

def ssim(x: np.ndarray, y: np.ndarray, on: str = "y", win_size: int = 11, k1: float = 0.01, k2: float = 0.03) -> float:
    assert x.shape[:2] == y.shape[:2]
    H, W = x.shape[0], x.shape[1]
    if H == 0 or W == 0:
        return 0.0
    X = _to_eval_space(np.clip(x, 0.0, 1.0), on)
    Y = _to_eval_space(np.clip(y, 0.0, 1.0), on)

    k = _safe_win_size(win_size, H, W)
    win = _gaussian_kernel1d(k, 1.5)

    l, c, s = _ssim_components(X, Y, k1, k2, win)
    val = float(np.mean(l * c * s))
    return max(0.0, min(1.0, val))

_MS_WEIGHTS = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float32)

def ms_ssim(
    x: np.ndarray,
    y: np.ndarray,
    on: str = "y",
    levels: int = 5,
    weights: np.ndarray | None = None,
    win_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    assert x.shape[:2] == y.shape[:2]
    H, W = x.shape[0], x.shape[1]
    if H == 0 or W == 0:
        return 0.0

    max_levels = int(levels)
    X = _to_eval_space(np.clip(x, 0.0, 1.0), on)
    Y = _to_eval_space(np.clip(y, 0.0, 1.0), on)

    vals = []
    cur = 0
    while True:
        Hs, Ws = X.shape[0], X.shape[1]
        if Hs == 0 or Ws == 0:
            break
        k = _safe_win_size(win_size, Hs, Ws)
        win = _gaussian_kernel1d(k, 1.5)

        l, c, s = _ssim_components(X, Y, k1, k2, win)
        if cur < max_levels - 1:
            vals.append(np.mean(c * s))
            X2 = _down2(X)
            Y2 = _down2(Y)
            if (X2.shape[0] == X.shape[0]) or (X2.shape[1] == X.shape[1]):
                cur += 1
                vals.append(np.mean(l * c * s))
                break
            X, Y = X2, Y2
            cur += 1
            continue
        else:
            vals.append(np.mean(l * c * s))
            break

    vals = np.array(vals, dtype=np.float64)
    if weights is None:
        w = _MS_WEIGHTS[:len(vals)].astype(np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32)[:len(vals)]
    w = w / (float(w.sum()) + 1e-12)

    out = float(np.prod(vals ** w))
    return max(0.0, min(1.0, out))