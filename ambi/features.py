import numpy as np

def entropy(x: np.ndarray) -> float:
    v = np.clip((x.flatten()*255.0).astype(np.int32), 0, 255)
    hist = np.bincount(v, minlength=256).astype(np.float64)
    p = hist / np.sum(hist)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

# def gradients(x: np.ndarray):
#     gy, gx = np.gradient(x)
#     gmag = np.sqrt(gx*gx + gy*gy)
#     return float(np.mean(gmag)), float(np.var(gmag))

def gradients(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]

    H, W = x.shape[:2]
    if H < 2 or W < 2:
        gmag = np.zeros_like(x, dtype=np.float32)
    else:
        gy, gx = np.gradient(x, edge_order=1)
        gmag = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    return float(gmag.mean()), float(gmag.var())

def laplacian_var(x: np.ndarray) -> float:
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    from numpy.lib.stride_tricks import sliding_window_view
    if x.ndim == 2:
        pad = np.pad(x, 1, mode="edge")
        sw = sliding_window_view(pad, (3,3))
        conv = np.tensordot(sw, k, axes=((2,3),(0,1)))
        return float(np.var(conv))
    return 0.0

def band_energies(y: np.ndarray):
    h, w = y.shape
    hh = h//2
    ww = w//2
    ll = y[:hh, :ww]
    lh = y[:hh, ww:]
    hl = y[hh:, :ww]
    hhb = y[hh:, ww:]
    e = lambda z: float(np.mean(z*z))
    return e(ll), e(lh), e(hl), e(hhb)

def neighbor_feats(q_left: int, q_top: int, berr: float):
    return float(q_left or 0), float(q_top or 0), float(berr or 0.0)

def preview_feats(N: int, s1: float, s1_s2: float, p_margin: float, topk_var: float):
    return float(N), float(s1), float(s1_s2), float(p_margin), float(topk_var)

# def block_features(block_ycc: np.ndarray, q_left=None, q_top=None, berr=None):
#     y = block_ycc[..., 0]
#     m = float(np.mean(y))
#     v = float(np.var(y))
#     g_mean, g_var = gradients(y)
#     lap = laplacian_var(y)
#     ent = entropy(y)
#     e_ll, e_lh, e_hl, e_hh = band_energies(y)
#     n_q_left, n_q_top, n_berr = neighbor_feats(q_left, q_top, berr)
#     pN, ps1, ps12, pm, pkv = preview_feats(1, 1.0, 0.0, 1.0, 0.0)
#     return np.array([
#         m, v, g_mean, g_var, lap, ent,
#         e_ll, e_lh, e_hl, e_hh,
#         n_q_left, n_q_top, n_berr,
#         pN, ps1, ps12, pm, pkv
#     ], dtype=np.float32)

def block_features(block_ycc: np.ndarray, q_left=None, q_top=None, berr=None):
    b = np.asarray(block_ycc, dtype=np.float32)
    if b.ndim != 3 or b.shape[-1] < 1:
        return np.zeros((18,), dtype=np.float32)

    y = np.clip(b[..., 0], 0.0, 1.0)
    H, W = y.shape[:2]

    def _safe(fn, default, *args, **kw):
        try:
            return fn(*args, **kw)
        except Exception:
            return default

    m  = float(np.mean(y)) if H and W else 0.0
    v  = float(np.var(y))  if H and W else 0.0

    g_mean, g_var = _safe(gradients, (0.0, 0.0), y)
    lap           = _safe(laplacian_var, 0.0, y)
    ent           = _safe(entropy, 0.0, y)

    e_ll, e_lh, e_hl, e_hh = _safe(band_energies, (0.0, 0.0, 0.0, 0.0), y)

    n_q_left, n_q_top, n_berr = _safe(
        neighbor_feats, (0.0, 0.0, 0.0), q_left, q_top, berr
    )

    pN, ps1, ps12, pm, pkv = _safe(
        preview_feats, (0.0, 0.0, 0.0, 0.0, 0.0),
        1, 1.0, 0.0, 1.0, 0.0
    )

    feats = np.array([
        m, v, g_mean, g_var, lap, ent,
        e_ll, e_lh, e_hl, e_hh,
        n_q_left, n_q_top, n_berr,
        pN, ps1, ps12, pm, pkv
    ], dtype=np.float32)

    if not np.all(np.isfinite(feats)):
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return feats