import argparse, json, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from scipy.ndimage import gaussian_filter

from ambi_core.bitstream import decode_image, encode_image, plan_image, tile_grid
from ambi_core.color import psnr_rgb, rgb_to_ycc
from ambi_core.metrics import ssim_y
from ambi_core.oracle import TileTable
from ambi_core.profile import Profile
from ambi_core.search import TileTables

# ------------------------------------------------------------------ metrics
_MS_W = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
_C1, _C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
MIN_SIDE = 11 * 2 ** 4          # five scales with an 11x11 window


def _blur(x):
    return gaussian_filter(x, 1.5, mode="nearest", truncate=3.5)[5:-5, 5:-5]      # 11 taps, 'valid' region


def _maps(a, b):
    """-> (mean contrast-structure, mean SSIM) of two float32 planes (0..255), Gaussian 11x11, sigma 1.5."""
    ma, mb = _blur(a), _blur(b)
    saa, sbb, sab = _blur(a * a) - ma * ma, _blur(b * b) - mb * mb, _blur(a * b) - ma * mb
    cs = (2 * sab + _C2) / (saa + sbb + _C2)
    lum = (2 * ma * mb + _C1) / (ma * ma + mb * mb + _C1)
    return float(cs.mean()), float((lum * cs).mean())


def ssim_gauss(a, b):
    return _maps(np.asarray(a, np.float32), np.asarray(b, np.float32))[1]


def ms_ssim(a, b):
    """Multi-scale SSIM (Wang et al. 2003, five scales) of two luma planes in 0..255. Needs both sides >= 176 px."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if min(a.shape) < MIN_SIDE:
        raise ValueError(f"MS-SSIM needs pictures of at least {MIN_SIDE} px per side (got {a.shape}); use --crop 256 or more")
    vals = []
    for j in range(5):
        cs, ss = _maps(a, b)
        if j < 4:
            vals.append(max(cs, 1e-6))
            h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
            a = a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))
            b = b[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))
        else:
            vals.append(max(ss, 1e-6))
    return float(np.prod(np.power(vals, _MS_W)))


METRICS = {"msssim": ms_ssim, "ssim": ssim_gauss}


def luma(rgb):
    return rgb_to_ycc(rgb)[0]


# ---------------------------------------------------------- map machinery
def scaled_tables(tables, lam0, lam_per_tile):
    """Cost tables whose distortion is multiplied by lam0/lam_t: the exact DP at lam0 then picks what it would pick at lam_t."""
    out = []
    for tb, lt in zip(tables, lam_per_tile):
        s = lam0 / float(lt)
        out.append(TileTable({k: (m, b, (d * s).astype(d.dtype), sse) for k, (m, b, d, sse) in tb.shapes.items()}))
    return out


def build_library(rgb, profile, tables, lams):
    """Encode the picture at every constant lambda; keep each tile's bits and luma reconstruction (before deblocking)."""
    cfg = profile.cfg
    T = cfg.tile
    ty, tx = tile_grid(*rgb.shape[:2], T)
    lib, ycc = {}, None
    for lam in lams:
        plan = plan_image(rgb, profile, lam, tables=tables)
        ycc = plan.ycc
        lib[lam] = dict(bits=np.array([r.total_bits for r in plan.results]),
                        y=[plan.recon[0, j * T:(j + 1) * T, i * T:(i + 1) * T].copy()
                           for j in range(ty) for i in range(tx)])
    return ycc, lib


def assemble(lib, row, ks, T, ty, tx):
    y = np.empty((ty * T, tx * T), np.float32)
    for t, k in enumerate(ks):
        j, i = divmod(t, tx)
        y[j * T:(j + 1) * T, i * T:(i + 1) * T] = lib[row[k]]["y"][t]
    return y


def map_bytes(lib, row, ks):
    return float(sum(lib[row[k]]["bits"][t] for t, k in enumerate(ks))) / 8


def descend(lib, row, ks, gamma, orig, shape, T, tx, metric, rng, max_pass=6):
    """Coordinate descent over tiles on  bytes - gamma * quality.  ks: index into `row` per tile (changed in place)."""
    H, W = shape
    ty = len(ks) // tx
    y = assemble(lib, row, ks, T, ty, tx)
    bits = np.array([lib[row[k]]["bits"][t] for t, k in enumerate(ks)], dtype=np.float64)
    q = metric(orig, y[:H, :W])
    cur = bits.sum() / 8 - gamma * q
    for _ in range(max_pass):
        changed = False
        for t in rng.permutation(len(ks)):
            j, i = divmod(int(t), tx)
            sl = (slice(j * T, (j + 1) * T), slice(i * T, (i + 1) * T))
            best = (cur, ks[t], q)
            for k in range(len(row)):
                if k == ks[t]:
                    continue
                y[sl] = lib[row[k]]["y"][t]
                qk = metric(orig, y[:H, :W])
                L = (bits.sum() - bits[t] + lib[row[k]]["bits"][t]) / 8 - gamma * qk
                if L < best[0] - 1e-9:
                    best = (L, k, qk)
            y[sl] = lib[row[best[1]]]["y"][t]
            if best[1] != ks[t]:
                ks[t] = best[1]; bits[t] = lib[row[best[1]]]["bits"][t]; cur, q = best[0], best[2]; changed = True
        if not changed:
            break
    return ks


def real_point(rgb, profile, lam, tables):
    """Encode for real, decode the bytes, measure: (lambda, bytes, PSNR, SSIM(Y), MS-SSIM(Y))."""
    r = encode_image(rgb, profile, lam, tables=tables)
    dec = decode_image(r.data, profile)
    return (lam, r.nbytes, psnr_rgb(rgb, dec), ssim_y(rgb, dec), ms_ssim(luma(rgb), luma(dec)))


def bytes_at(points, key, value):
    """Interpolated size (log scale) at which the curve reaches `value` of metric `key`; None outside its range."""
    x = np.array([p[key] for p in points]); lb = np.log([p[1] for p in points])
    o = np.argsort(x, kind="stable"); x, lb = x[o], lb[o]
    if value < x[0] or value > x[-1]:
        return None
    return float(np.exp(np.interp(value, x, lb)))


# ------------------------------------------------------------------- main
def probe_picture(rgb, profile, op_lams, mults, gammas, metric_name, seed, log=print, tabs=None):
    cfg = profile.cfg
    T = cfg.tile
    H, W = rgb.shape[:2]
    ty, tx = tile_grid(H, W, T)
    metric = METRICS[metric_name]
    tabs = tabs or TileTables(rgb, profile)       # callers that need the tables too (the side-by-side sheet) pass them in
    if not tabs.full:
        raise SystemExit("picture too large for full cost tables; use a smaller --crop")
    k1 = list(mults).index(1.0)
    rows = {lam0: [round(lam0 * m, 6) for m in mults] for lam0 in op_lams}
    all_lams = sorted({l for r in rows.values() for l in r})
    ycc, lib = build_library(rgb, profile, tabs.tables, all_lams)
    orig = ycc[0][:H, :W]
    const_curve = [real_point(rgb, profile, lam, tabs.tables) for lam in all_lams]      # real files, constant lambda
    out = dict(const=const_curve, ops=[])
    stds = np.array([orig[(t // tx) * T:(t // tx + 1) * T, (t % tx) * T:(t % tx + 1) * T].std() for t in range(ty * tx)])
    rng = np.random.default_rng(seed)
    for lam0 in op_lams:
        row = rows[lam0]
        const_ks = [k1] * (ty * tx)
        B = [map_bytes(lib, row, [k] * (ty * tx)) for k in range(len(row))]
        Q = [metric(orig, assemble(lib, row, [k] * (ty * tx), T, ty, tx)[:H, :W]) for k in range(len(row))]
        dq = Q[k1 + 1] - Q[k1 - 1]
        gamma_ref = (B[k1 + 1] - B[k1 - 1]) / dq if abs(dq) > 1e-9 else None       # both differences are negative: bytes per unit quality
        op = dict(lam0=lam0, gamma_ref=gamma_ref, maps=[])
        if gamma_ref is None or gamma_ref <= 0:
            log(f"    lambda {lam0}: quality does not change with lambda here; skipped")
            out["ops"].append(op)
            continue
        for gf in gammas:
            ks = list(const_ks)
            descend(lib, row, ks, gf * gamma_ref, orig, (H, W), T, tx, metric, rng)
            lam_t = [row[k] for k in ks]
            pt = real_point(rgb, profile, lam0, scaled_tables(tabs.tables, lam0, lam_t))
            op["maps"].append(dict(gamma_factor=gf, ks=[int(k) for k in ks], real=pt,
                                   log2m=[float(np.log2(mults[k])) for k in ks]))
        op["stds"] = stds.tolist()
        out["ops"].append(op)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--profile", required=True, help="a profile WITHOUT intra prediction")
    ap.add_argument("--crop", type=int, default=256, help="crop side in px (multiple of 64, at least 192); 0 = whole pictures")
    ap.add_argument("--crops-per-image", type=int, default=1)
    ap.add_argument("--max-images", type=int, default=6)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.03, 0.08], help="global operating points to improve on")
    ap.add_argument("--mults", type=float, nargs="+", default=[0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0],
                    help="per-tile lambda = global lambda x one of these (must include 1)")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.25, 1.6],
                    help="exchange rates to try, relative to the constant-lambda curve's slope")
    ap.add_argument("--metric", choices=sorted(METRICS), default="msssim", help="quality the map is optimised for")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args()
    if 1.0 not in a.mults:
        sys.exit("--mults must include 1")
    mults = sorted(a.mults)
    profile = Profile.load(a.profile)
    if profile.cfg.intra_preds:
        sys.exit("this probe needs a profile without intra prediction (tiles must be independent)")
    from side_by_side import load_rgb
    files = [f for f in sorted(pathlib.Path(a.image_dir).rglob("*"))
             if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")]
    if a.max_images:
        files = files[:a.max_images]
    rng = np.random.default_rng(a.seed)
    per_pic, t0 = [], time.time()
    for fi, f in enumerate(files):
        full, _ = load_rgb(f)
        H, W = full.shape[:2]
        if a.crop and (H < a.crop or W < a.crop):
            continue
        jobs = [(0, 0, full)] if not a.crop else []
        for _ in range(a.crops_per_image if a.crop else 0):
            y = int(rng.integers(0, (H - a.crop) // 64 + 1)) * 64
            x = int(rng.integers(0, (W - a.crop) // 64 + 1)) * 64
            jobs.append((y, x, np.ascontiguousarray(full[y:y + a.crop, x:x + a.crop])))
        for (y, x, rgb) in jobs:
            r = probe_picture(rgb, profile, a.lams, mults, a.gammas, a.metric, a.seed + fi)
            r.update(image=f.name, y=y, x=x)
            per_pic.append(r)
        print(f"  {fi + 1}/{len(files)} {f.name}  ({time.time() - t0:.0f}s)", flush=True)

    keys = ((4, "MS-SSIM"), (3, "SSIM(Y)"), (2, "PSNR"))
    print(f"\nlambda MAP vs constant lambda, REAL files at equal quality; optimised for {a.metric}; {len(per_pic)} pictures")
    print("(negative = the map's file is smaller than the constant-lambda file of the same quality)")
    best = {}
    for lam0 in a.lams:
        print(f"\nglobal lambda {lam0}")
        print(f"  {'gamma x':>8} {'pics':>5} " + " ".join(f"{n:>9}" for _, n in keys) + "   mean |log2 mult|")
        for gi, gf in enumerate(a.gammas):
            cells, n_pics, spread = [], 0, []
            for key, _ in keys:
                mb = cb = 0.0; n = 0
                for p in per_pic:
                    op = next(o for o in p["ops"] if o["lam0"] == lam0)
                    if not op["maps"]:
                        continue
                    m = op["maps"][gi]
                    cbytes = bytes_at(p["const"], key, m["real"][key])
                    if cbytes is None:
                        continue
                    mb += m["real"][1]; cb += cbytes; n += 1
                    if key == 4:
                        spread.append(np.mean(np.abs(m["log2m"])))
                cells.append(f"{100 * (mb / cb - 1):>+8.1f}%" if n else f"{'n/a':>9}")
                if key == 4:
                    n_pics = n
                    if n:
                        best[(lam0, gf)] = 100 * (mb / cb - 1)
            print(f"  {gf:>8g} {n_pics:>5} " + " ".join(cells) + f"   {np.mean(spread) if spread else float('nan'):.2f}")
    # does the search give more quality to flat or to busy tiles?
    lo, hi = [], []
    for p in per_pic:
        for op in p["ops"]:
            for m in op["maps"]:
                if m["gamma_factor"] != 1.0:
                    continue
                s = np.array(op["stds"]); lm = np.array(m["log2m"]); med = np.median(s)
                lo += lm[s <= med].tolist(); hi += lm[s > med].tolist()
    if lo and hi:
        print(f"\nmean log2(lambda multiplier) at gamma x 1: flatter half of the tiles {np.mean(lo):+.2f}, busier half {np.mean(hi):+.2f}"
              "  (positive = coarser than the global lambda)")
    if best:
        top = min(best, key=best.get)
        print(f"\nbest cell (picked after seeing the results, so optimistic): global lambda {top[0]}, gamma x {top[1]}: {best[top]:+.1f}% at equal MS-SSIM")
        print("rule of thumb: if the saving is smaller than about 2% the map is not worth building (and this is an upper bound: "
              "the map was chosen knowing the original picture).")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(per_pic, indent=1))


if __name__ == "__main__":
    main()
