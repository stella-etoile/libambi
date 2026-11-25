import argparse, io, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import Image
from ambi_core import psnr_rgb
from ambi_core.bitstream import decode_image, encode_image
from ambi_core.metrics import ssim_y
from ambi_core.profile import Profile
from ambi_core.search import TileTables
from side_by_side import load_rgb

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--profile", default="profiles/photo-demo.ambiprofile")
    ap.add_argument("--vs", nargs="+", choices=["png", "jpeg", "webp", "both", "all"], default=["png", "jpeg"],
                    help="what to compare against (default: png jpeg)")
    ap.add_argument("--policy", help="trained .ambipolicy (fast partitioner)")
    ap.add_argument("--oracle", action="store_true", help="use the exact oracle instead (slow)")
    ap.add_argument("--prune-k", type=int, default=0, metavar="K",
                    help="with --policy: exact search over the policy's K best actions per rectangle (3 is a good start; 0 = the policy alone)")
    ap.add_argument("--workers", type=int, default=1, metavar="N",
                    help="with --prune-k: N processes price the candidate leaves while the policy runs on --device (same result, much faster)")
    ap.add_argument("--crop", type=int, default=256, help="crop side in px (multiple of 64); 0 = whole image")
    ap.add_argument("--crops-per-image", type=int, default=2)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args()
    if not a.policy and not a.oracle:
        sys.exit("give --policy FILE or --oracle")
    if a.prune_k and not a.policy:
        sys.exit("--prune-k needs --policy")
    if a.workers > 1 and not a.prune_k:
        sys.exit("--workers needs --prune-k (it prices the pruned search's leaves in parallel)")
    VS = set()
    for v in a.vs:
        VS |= {"png", "webp"} if v == "both" else {"png", "jpeg", "webp"} if v == "all" else {v}
    prof = Profile.load(a.profile)
    part = None
    if a.policy and not a.oracle:
        from ambi_rl.pruned import make_partitioner
        part = make_partitioner(a.policy, prof, a.device, a.prune_k, a.workers)
    rng = np.random.default_rng(a.seed)
    QS = {"webp": list(range(5, 100, 5)) + [98, 100],
          "jpeg": [2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 92, 94, 96, 98, 100]}
    LABEL = {"webp": "WebP", "jpeg": "JPEG"}


    def base_curve(rgb, name):
        pts = []
        for q in QS[name]:
            buf = io.BytesIO()
            if name == "webp":
                Image.fromarray(rgb).save(buf, "WEBP", quality=q, method=4)
            else:
                Image.fromarray(rgb).save(buf, "JPEG", quality=q, subsampling="4:2:0", optimize=True)
            dec = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
            pts.append((psnr_rgb(rgb, dec), ssim_y(rgb, dec), len(buf.getvalue())))
        return {"psnr": np.array([p[0] for p in pts]), "ssim": np.array([p[1] for p in pts]),
                "logb": np.log([p[2] for p in pts])}


    def bytes_at(curve, key, value):
        x, lb = curve[key], curve["logb"]
        o = np.argsort(x, kind="stable")
        x, lb = x[o], lb[o]
        if value < x[0] or value > x[-1]:
            return None
        return float(np.exp(np.interp(value, x, lb)))


    files = [f for f in sorted(pathlib.Path(a.image_dir).rglob("*")) if f.suffix.lower() in (".webp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")]
    if a.max_images:
        files = files[:a.max_images]
    rows = []
    t_all = time.time()
    for fi, f in enumerate(files):
        full, _note = load_rgb(f); H, W = full.shape[:2]
        if a.crop and (H < a.crop or W < a.crop):
            continue
        jobs = [(0, 0, full)] if not a.crop else []
        for _ in range(a.crops_per_image if a.crop else 0):
            y = int(rng.integers(0, (H - a.crop) // 64 + 1)) * 64
            x = int(rng.integers(0, (W - a.crop) // 64 + 1)) * 64
            jobs.append((y, x, np.ascontiguousarray(full[y:y + a.crop, x:x + a.crop])))
        for (y, x, rgb) in jobs:
            curves = {n: base_curve(rgb, n) for n in ("jpeg", "webp") if n in VS}
            buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "PNG", optimize=True, compress_level=9)
            png_bytes = len(buf.getvalue())
            tabs = TileTables(rgb, prof) if a.oracle else None
            for lam in a.lams:
                r = encode_image(rgb, prof, lam, tables=tabs.tables) if a.oracle else encode_image(rgb, prof, lam, partitioner=part)
                dec = decode_image(r.data, prof)
                ps = psnr_rgb(rgb, dec)
                ss = ssim_y(rgb, dec)
                rows.append(dict(image=f.name, y=y, x=x, h=rgb.shape[0], w=rgb.shape[1], lam=lam, ambi_bytes=r.nbytes,
                                 psnr=ps, ssim=ss,
                                 **{f"{n}_bytes_same_{k}": bytes_at(c, k, v) for n, c in curves.items() for k, v in (("psnr", ps), ("ssim", ss))},
                                 png_bytes=png_bytes, orig_file_bytes=f.stat().st_size if not a.crop else None))
        print(f"  {fi + 1}/{len(files)} {f.name}  ({time.time() - t_all:.0f}s)", flush=True)

    who = "oracle" if a.oracle else "policy " + pathlib.Path(a.policy).name + (f" + exact search over its top {a.prune_k}" if a.prune_k else "")
    if a.prune_k and part is not None:
        print(part.summary(), flush=True)
        part.close()
    summary = {}


    def S(lam):
        return summary.setdefault(str(lam), {})


    if "png" in VS:
        print(f"\nAMBI ({who}) against the same pixels stored as lossless PNG")
        print(f"  {'lambda':>7} {'crops':>6} {'PSNR dB':>8} {'SSIM':>7} {'AMBI bpp':>9} {'PNG bpp':>8} {'AMBI/PNG':>9} {'times smaller':>14}")
        for lam in a.lams:
            rs = [r for r in rows if r["lam"] == lam]
            if not rs:
                continue
            px = sum(r["h"] * r["w"] for r in rs)
            ab, pb = sum(r["ambi_bytes"] for r in rs), sum(r["png_bytes"] for r in rs)
            per = np.array([r["ambi_bytes"] / r["png_bytes"] for r in rs])
            S(lam).update(n=len(rs), psnr=float(np.mean([r["psnr"] for r in rs])), ssim=float(np.mean([r["ssim"] for r in rs])),
                          ambi_bpp=ab * 8 / px, png_bpp=pb * 8 / px, ambi_over_png=ab / pb, median_ambi_over_png=float(np.median(per)))
            print(f"  {lam:>7g} {len(rs):>6} {np.mean([r['psnr'] for r in rs]):>8.2f} {np.mean([r['ssim'] for r in rs]):>7.4f} "
                  f"{ab * 8 / px:>9.2f} {pb * 8 / px:>8.2f} {100 * ab / pb:>8.1f}% {pb / ab:>13.1f}x")
        print("AMBI/PNG = total AMBI bytes / total PNG bytes over all crops (pooled); SSIM is on luma, 1.0 = identical.")
        print("PNG keeps every pixel and AMBI does not: read the saving together with the PSNR / SSIM it reaches.")


    def base_table(name, key, label):
        col = f"{name}_bytes_same_{key}"
        nm = LABEL[name]
        print(f"\nAMBI ({who}) vs {nm} re-encoded to the same {label}; positive = AMBI file bigger, negative = AMBI smaller")
        print(f"  {'lambda':>7} {'crops':>6} {'PSNR dB':>8} {'SSIM':>7} {'bpp':>6} {'pooled':>8} {'median':>8} {'smaller in':>11}")
        for lam in a.lams:
            rs = [r for r in rows if r["lam"] == lam and r.get(col)]
            if not rs:
                print(f"  {lam:>7g}   (no images inside {nm}'s {label} range)")
                continue
            pooled = sum(r["ambi_bytes"] for r in rs) / sum(r[col] for r in rs) - 1
            per = np.array([r["ambi_bytes"] / r[col] - 1 for r in rs])
            bpp = sum(r["ambi_bytes"] for r in rs) * 8 / sum(r["h"] * r["w"] for r in rs)
            S(lam).update({f"{name}_{key}_n": len(rs), f"{name}_{key}_pooled_pct": 100 * pooled,
                           f"{name}_{key}_median_pct": 100 * float(np.median(per)), f"{name}_{key}_frac_smaller": float((per < 0).mean())})
            print(f"  {lam:>7g} {len(rs):>6} {np.mean([r['psnr'] for r in rs]):>8.2f} {np.mean([r['ssim'] for r in rs]):>7.4f} "
                  f"{bpp:>6.2f} {100 * pooled:>+7.1f}% {100 * float(np.median(per)):>+7.1f}% {100 * float((per < 0).mean()):>9.0f}%")
        print(f"pooled = total AMBI bytes / total equal-quality {nm} bytes; median = median over images; 'smaller in' = share of images where AMBI wins")


    for name in ("jpeg", "webp"):
        if name in VS:
            base_table(name, "psnr", "PSNR")
            base_table(name, "ssim", "SSIM(Y)")

    if not a.crop:
        print("\nvs the original image files on disk (their quality is the reference, so AMBI is lossy against them):")
        for lam in a.lams:
            rs = [r for r in rows if r["lam"] == lam]
            if rs:
                print(f"  lambda {lam:g}: AMBI = {100 * sum(r['ambi_bytes'] for r in rs) / sum(r['orig_file_bytes'] for r in rs):.0f}% of the original file size "
                      f"at {np.mean([r['psnr'] for r in rs]):.1f} dB vs the decoded original")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
