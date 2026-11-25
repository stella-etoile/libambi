import argparse, json, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import numpy as np
from PIL import Image, ImageDraw

from ambi_core.bitstream import decode_image, encode_image, tile_grid
from ambi_core.color import psnr_rgb
from ambi_core.metrics import ssim_y
from ambi_core.profile import Profile
from ambi_core.search import TileTables
import lambda_map_probe as lmp
from side_by_side import EXTS, HEAD_AMBI, HEAD_PNG, MIN_PANEL_W, compose, kb, load_font, load_rgb, png_size

HEAD_MAP = (110, 60, 150)
LABELS = {"msssim": "MS-SSIM(Y)", "ssim": "SSIM(Y)", "psnr": "PSNR"}
FINER, COARSER = (30, 90, 200), (200, 80, 20)        # diverging pair: blue = finer than the global lambda, orange = coarser


# ------------------------------------------------------------ real files
def measure(rgb, dec):
    y0, y1 = lmp.luma(rgb), lmp.luma(dec)
    ms = lmp.ms_ssim(y0, y1) if min(y0.shape) >= lmp.MIN_SIDE else None
    return dict(psnr=psnr_rgb(rgb, dec), ssim=ssim_y(rgb, dec), msssim=ms)


def encode_real(rgb, profile, lam, tables):
    """Encode for real, decode the bytes, measure."""
    r = encode_image(rgb, profile, lam, tables=tables)
    dec = decode_image(r.data, profile)
    return dict(lam=lam, data=r.data, nbytes=r.nbytes, dec=dec, **measure(rgb, dec))


def map_lambdas(lam0, log2m):
    """Per-tile lambdas, rounded exactly as lambda_map_probe does (so a --map-json file is reproduced bit for bit)."""
    return [round(lam0 * 2.0 ** float(v), 6) for v in log2m]


def match_constant(rgb, profile, tables, target, key, lo, hi, iters=14):
    """Real constant-lambda file with quality `key` >= target and the largest lambda in [lo, hi] that still does that.
    Returns (encode, ok): ok is False when the target lies outside what [lo, hi] can reach (the nearest end is returned)."""
    a = encode_real(rgb, profile, lo, tables)
    if a[key] < target:
        return a, False
    b = encode_real(rgb, profile, hi, tables)
    if b[key] >= target:
        return b, False
    best, L, H = a, np.log(lo), np.log(hi)
    for _ in range(iters):
        e = encode_real(rgb, profile, float(np.exp((L + H) / 2)), tables)
        if e[key] >= target:
            best, L = e, np.log(e["lam"])
        else:
            H = np.log(e["lam"])
    return best, True


# ---------------------------------------------------------------- panels
def multiplier_overlay(rgb, log2m, T, tx, S, span=2.0, labels=True):
    """The picture, enlarged S times, with every tile tinted by its lambda multiplier and (labels=True) outlined and labelled with it."""
    H, W = rgb.shape[:2]
    base = Image.fromarray(rgb).convert("RGBA").resize((W * S, H * S), Image.NEAREST)
    tint = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(tint)
    for t, v in enumerate(log2m):
        j, i = divmod(t, tx)
        box = [i * T * S, j * T * S, min((i + 1) * T, W) * S - 1, min((j + 1) * T, H) * S - 1]
        a = min(abs(float(v)) / span, 1.0)
        d.rectangle(box, fill=(COARSER if v > 0 else FINER) + ((int(35 + 150 * a) if abs(v) > 1e-9 else 0),))
    out = Image.alpha_composite(base, tint)
    if not labels:
        return np.array(out.convert("RGB"))
    fs = max(12, 10 * S)
    d, f = ImageDraw.Draw(out), load_font(fs)
    for t, v in enumerate(log2m):
        j, i = divmod(t, tx)
        x0, y0, x1, y1 = i * T * S, j * T * S, min((i + 1) * T, W) * S, min((j + 1) * T, H) * S
        d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 255, 255, 150))
        txt = f"x{2.0 ** float(v):.2g}"
        d.text(((x0 + x1 - f.getlength(txt)) / 2, (y0 + y1) / 2 - fs / 2), txt, font=f, fill=(255, 255, 255),
               stroke_width=2, stroke_fill=(0, 0, 0))
    return np.array(out.convert("RGB"))


def difference_picture(a, b, gain=4):
    """White where the two decoded pictures agree, darker where they differ (max over channels, magnified `gain` times)."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    return np.repeat(np.clip(255 - gain * d, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2), float(d.mean())


def quality_lines(q, nbytes, px, key):
    lines = [(f"size: {kb(nbytes)}", True), (f"{nbytes * 8 / px:.3f} bits per pixel", False),
             (f"PSNR {q['psnr']:.2f} dB    SSIM(Y) {q['ssim']:.4f}", key != "msssim")]
    lines.append((f"MS-SSIM(Y) {q['msssim']:.4f}" if q["msssim"] is not None else "MS-SSIM(Y): crop too small", key == "msssim"))
    return lines


# ----------------------------------------------------------------- sheet
def build_sheet(rgb, name, crop_at, profile, tabs, lam0, gamma_factor, log2m, match, metric, zoom=0, zoom_at=None, cols=3):
    """Encode the constant and the map files for real and draw the sheet. -> dict(canvas, const, map, ...)."""
    cfg = profile.cfg
    T = cfg.tile
    h, w = rgb.shape[:2]
    ty, tx = tile_grid(h, w, T)
    px = h * w
    const0 = encode_real(rgb, profile, lam0, tabs.tables)
    mp = encode_real(rgb, profile, lam0, lmp.scaled_tables(tabs.tables, lam0, map_lambdas(lam0, log2m)))
    key = match if mp[match] is not None else "ssim"
    cm, ok = match_constant(rgb, profile, tabs.tables, mp[key], key, lam0 / 8, lam0 * 8)
    delta = 100 * (mp["nbytes"] / cm["nbytes"] - 1)

    S = max(1, -(-MIN_PANEL_W // w))
    tile_std = np.array([lmp.luma(rgb)[(t // tx) * T:(t // tx + 1) * T, (t % tx) * T:(t % tx + 1) * T].std() for t in range(ty * tx)])
    lm = np.asarray(log2m, dtype=float)
    flat, busy = lm[tile_std <= np.median(tile_std)], lm[tile_std > np.median(tile_std)]
    overlay = multiplier_overlay(rgb, lm, T, tx, S)
    native = multiplier_overlay(rgb, lm, T, tx, 1, labels=False)          # what the zoom patch is cut from
    diff, mad = difference_picture(cm["dec"], mp["dec"])
    png_b = png_size(rgb)

    panels = [
        dict(head="Original (crop)", head_color=HEAD_PNG, rgb=rgb,
             lines=[(name, True), (f"crop x={crop_at[0]} y={crop_at[1]}, {w}x{h} px", False),
                    (f"lossless PNG of the crop: {kb(png_b)}", False), ("", False), ("", False)]),
        dict(head=f"Constant lambda {cm['lam']:.4g}", head_color=HEAD_AMBI, rgb=cm["dec"],
             lines=quality_lines(cm, cm["nbytes"], px, key) +
                   [(f"same {LABELS[key]} as the map" + ("" if ok else " (limit of the lambda range)"), False)]),
        dict(head=f"lambda MAP  (from lambda {lam0:g})", head_color=HEAD_MAP, rgb=mp["dec"],
             lines=quality_lines(mp, mp["nbytes"], px, key) +
                   [(f"{delta:+.1f}% size vs the constant file of the same {LABELS[key]}", True)]),
        dict(head=f"Constant lambda {lam0:g}  (the map's start)", head_color=HEAD_AMBI, rgb=const0["dec"],
             lines=quality_lines(const0, const0["nbytes"], px, key) + [("the global lambda the map was built around", False)]),
        dict(head="Where the map changed lambda", head_color=HEAD_MAP, rgb=overlay, zoom_rgb=native,
             lines=[("blue = finer (smaller lambda), orange = coarser", False),
                    (f"mean |log2 multiplier| {np.abs(lm).mean():.2f}", False),
                    (f"flatter half of the tiles {flat.mean():+.2f}, busier half {busy.mean():+.2f}" if len(flat) and len(busy) else "", False),
                    (f"optimised for {LABELS[metric]}, gamma x {gamma_factor:g}", False)]),
        dict(head="Difference of the two files (x4)", head_color=HEAD_PNG, rgb=diff,
             lines=[("constant (same quality) vs map", False), ("dark = the pixels differ", False),
                    (f"mean |difference| {mad:.2f} of 255", False)]),
    ]
    zs = zoom
    zpos = tuple(zoom_at) if zoom_at else ((w - zs) // 2, (h - zs) // 2) if zs else (0, 0)
    title = [f"{profile.name}: one constant lambda vs a per-tile lambda MAP    ({name}, crop x={crop_at[0]} y={crop_at[1]})",
             f"all pictures are decoded real .ambi files; the map was searched knowing the original picture (an upper bound for a "
             f"learned map), global lambda {lam0:g}",
             f"the constant file in the first row was searched until its {LABELS[key]} equals the map's: at equal quality "
             f"the map file is {abs(delta):.1f}% {'smaller' if delta < 0 else 'bigger'}"]
    canvas = compose(panels, title, bool(zs), zs, zpos, cols)
    return dict(canvas=canvas, const=cm, map=mp, const0=const0, key=key, delta=delta, matched=ok)


# ------------------------------------------------------------------ main
def crops_of(files, crop, crops_per_image, seed):
    """(file index, file, y, x, rgb crop) exactly as lambda_map_probe.py picks them (same seed = same crops)."""
    rng = np.random.default_rng(seed)
    for fi, f in enumerate(files):
        full, _ = load_rgb(f)
        H, W = full.shape[:2]
        if crop and (H < crop or W < crop):
            continue
        if not crop:
            yield fi, f, 0, 0, full
            continue
        for _ in range(crops_per_image):
            y = int(rng.integers(0, (H - crop) // 64 + 1)) * 64
            x = int(rng.integers(0, (W - crop) // 64 + 1)) * 64
            yield fi, f, y, x, np.ascontiguousarray(full[y:y + crop, x:x + crop])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--profile", required=True, help="a profile WITHOUT intra prediction")
    ap.add_argument("--lam", type=float, nargs="+", help="global lambda(s) the map is built around (default 0.02; with --map-json: every one in the file)")
    ap.add_argument("--gamma", type=float, nargs="+", default=[1.0], help="exchange rate(s) of the search, relative to the constant curve's slope")
    ap.add_argument("--metric", choices=sorted(lmp.METRICS), default="msssim", help="quality the map is optimised for")
    ap.add_argument("--match", choices=sorted(LABELS), help="quality the two files are matched on (default: the --metric)")
    ap.add_argument("--map-json", help="use the maps in this lambda_map_probe.py --json file instead of searching")
    ap.add_argument("--crop", type=int, default=256, help="crop side in px (multiple of 64, at least 192); 0 = whole pictures")
    ap.add_argument("--crops-per-image", type=int, default=1)
    ap.add_argument("--max-images", type=int, default=3)
    ap.add_argument("--mults", type=float, nargs="+", default=[0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zoom", type=int, default=0, help="also show a magnified patch of this size (px) under each panel")
    ap.add_argument("--zoom-at", type=int, nargs=2, metavar=("X", "Y"), help="top-left corner of the zoom patch (default: centre)")
    ap.add_argument("--cols", type=int, default=3, help="panels per row")
    ap.add_argument("--analysis-dir", help="results go to ANALYSIS\\<name>_lambdamap and ANALYSIS\\<name>_lambdamap_files "
                                           "(default: <the picture folder>_analysis)")
    a = ap.parse_args()
    if 1.0 not in a.mults:
        sys.exit("--mults must include 1")
    mults = sorted(a.mults)
    profile = Profile.load(a.profile)
    if profile.cfg.intra_preds:
        sys.exit("this needs a profile without intra prediction (tiles must be independent)")
    match = a.match or a.metric
    src_dir = pathlib.Path(a.image_dir)
    files = sorted(f for f in src_dir.rglob("*") if f.suffix.lower() in EXTS)
    if not files:
        sys.exit(f"no images in {a.image_dir}")
    base = pathlib.Path(a.analysis_dir) if a.analysis_dir else src_dir.with_name(src_dir.name + "_analysis")
    prefix = base.name[:-len("_analysis")] if base.name.endswith("_analysis") else base.name
    cmp_dir, conv = base / f"{prefix}_lambdamap", base / f"{prefix}_lambdamap_files"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    conv.mkdir(parents=True, exist_ok=True)

    saved = None
    if a.map_json:
        saved = json.loads(pathlib.Path(a.map_json).read_text())
        by_name = {f.name: f for f in files}
        jobs = []
        for e in saved:
            f = by_name.get(e["image"])
            if f is None:
                print(f"skip {e['image']}: not found under {a.image_dir}")
                continue
            n = len(e["ops"][0]["maps"][0]["log2m"]) if e["ops"] and e["ops"][0]["maps"] else 0
            side = int(round(np.sqrt(n))) * profile.cfg.tile
            full, _ = load_rgb(f)
            jobs.append((0, f, e["y"], e["x"], np.ascontiguousarray(full[e["y"]:e["y"] + side, e["x"]:e["x"] + side]), e))
        jobs = jobs[:a.max_images] if a.max_images else jobs
    else:
        picked = files[:a.max_images] if a.max_images else files
        jobs = [(fi, f, y, x, rgb, None) for fi, f, y, x, rgb in crops_of(picked, a.crop, a.crops_per_image, a.seed)]
    if not jobs:
        sys.exit("nothing to draw (no crop fits, or the json names no image found in --image-dir)")

    tot_map = tot_const = 0
    t0 = time.time()
    for fi, f, y, x, rgb, entry in jobs:
        tabs = TileTables(rgb, profile)
        if not tabs.full:
            print(f"skip {f.name}: crop too large for full cost tables; use a smaller --crop")
            continue
        if entry is None:
            res = lmp.probe_picture(rgb, profile, a.lam or [0.02], mults, a.gamma, a.metric, a.seed + fi, tabs=tabs)
            ops = res["ops"]
        else:
            ops = entry["ops"]
        for op in ops:
            if entry is not None and a.lam and not any(abs(op["lam0"] - l) < 1e-9 for l in a.lam):
                continue
            wanted = [m for m in op["maps"] if any(abs(m["gamma_factor"] - g) < 1e-9 for g in a.gamma)]
            if op["maps"] and not wanted:
                print(f"  {f.name} lambda {op['lam0']:g}: no map with gamma x {a.gamma}; the file has "
                      f"{sorted({m['gamma_factor'] for m in op['maps']})} (use --gamma)")
            for m in wanted:
                r = build_sheet(rgb, f.name, (x, y), profile, tabs, op["lam0"], m["gamma_factor"], m["log2m"], match, a.metric,
                                a.zoom, a.zoom_at, a.cols)
                stem = f"{f.stem[:20]}_{f.suffix.lower().lstrip('.')}_crop{x}_{y}_lam{op['lam0']:g}_g{m['gamma_factor']:g}"
                r["canvas"].save(cmp_dir / f"{stem}.png")
                (conv / f"{stem}_map.ambi").write_bytes(r["map"]["data"])
                (conv / f"{stem}_constant_same_quality.ambi").write_bytes(r["const"]["data"])
                Image.fromarray(rgb).save(conv / f"{stem}_original.png")
                tot_map += r["map"]["nbytes"]; tot_const += r["const"]["nbytes"]
                print(f"  {f.name} crop ({x},{y}) lambda {op['lam0']:g} gamma x{m['gamma_factor']:g}: map {r['map']['nbytes']:,} B vs "
                      f"constant {r['const']['nbytes']:,} B at equal {LABELS[r['key']]} ({r['delta']:+.1f}%)"
                      f"{'' if r['matched'] else '  [match at the end of the lambda range]'}   ({time.time() - t0:.0f}s)", flush=True)
    if tot_const:
        print(f"\nall sheets pooled: the map files are {100 * (tot_map / tot_const - 1):+.1f}% vs constant files of equal "
              f"{LABELS[match]} (negative = smaller)")
        print(f"sheets: {cmp_dir}\nfiles (.ambi, originals): {conv}")


if __name__ == "__main__":
    main()
