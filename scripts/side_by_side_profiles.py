import argparse, json, pathlib, re, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import Image
from ambi_core import psnr_rgb
from ambi_core.bitstream import decode_image, encode_image
from ambi_core.metrics import ssim_y
from ambi_core.profile import Profile
from ambi_core.progress import LiveLine, fmt_time
from ambi_core.search import TileTables
from ambi_core.parallel import run_tasks
from compare_profiles import Bar, QueueBar, drain, init_worker, load_profile_cached, work_units
from side_by_side import (EXTS, HEAD_JPEG, HEAD_PNG, compose, jpeg_bytes, jpeg_decode, jpeg_match_psnr, jpeg_match_size, kb,
                          load_rgb, png_size)

PROFILE_COLORS = [(30, 90, 170), (20, 125, 115), (105, 65, 155), (165, 55, 95), (95, 110, 30), (40, 40, 140)]


def safe(label):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-") or "profile"


def pick_crops(files, crop, crops_per_image, seed):
    rng = np.random.default_rng(seed)
    out = []
    for f in files:
        full, note = load_rgb(f)
        H, W = full.shape[:2]
        if not crop:
            out.append((f, 0, 0, full, note, None))
            continue
        if H < crop or W < crop:
            print(f"skip {f.name}: smaller than the crop ({W}x{H})")
            continue
        for _ in range(crops_per_image):
            y = int(rng.integers(0, (H - crop) // 64 + 1)) * 64
            x = int(rng.integers(0, (W - crop) // 64 + 1)) * 64
            out.append((f, y, x, np.ascontiguousarray(full[y:y + crop, x:x + crop]), note, (x, y)))
    return out


def encode_profile(rgb, prof, label, lams, oracle, part, bar, conv, stem):
    tabs = None
    ntab, w_tile, w_enc = work_units(rgb, prof, oracle, len(lams))
    if oracle:
        tabs = TileTables(rgb, prof, progress=lambda n, tot: bar.render(f"{label} tables {n}/{tot}", n * w_tile))
        bar.done += ntab * w_tile
    out = []
    for lam in lams:
        bar.render(f"{label} encode lambda {lam:g}")
        t0 = time.time()
        r = (encode_image(rgb, prof, lam, tables=tabs.tables if tabs.full else None) if oracle
             else encode_image(rgb, prof, lam, partitioner=part))
        t_enc = time.time() - t0
        path = conv / f"{stem}_{safe(label)}_lam{lam:g}.ambi"
        path.write_bytes(r.data)
        t0 = time.time()
        dec = decode_image(path.read_bytes(), prof)
        t_dec = time.time() - t0
        out.append(dict(lam=lam, nb=len(r.data), dec=dec, psnr=psnr_rgb(rgb, dec), ssim=ssim_y(rgb, dec),
                        t_enc=t_enc, t_dec=t_dec, deblock=r.deblock, path=path))
        bar.done += w_enc
    return out


def stem_of(f, crop_at):
    return f"{f.stem[:20]}_{f.suffix.lower().lstrip('.')}" + (f"_crop{crop_at[0]}_{crop_at[1]}" if crop_at else "")


def encode_task(args):
    k, pi, rgb, path, label, lams, conv, stem = args
    t0 = time.time()
    return k, pi, encode_profile(rgb, load_profile_cached(path), label, lams, True, None, QueueBar((k, pi)), conv, stem), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image-dir", help="pictures in this folder (sorted; the first --max-images are used)")
    src.add_argument("--image", help="one picture")
    ap.add_argument("--profiles", nargs="+", required=True, metavar="FILE", help="two or more .ambiprofile files, left to right")
    ap.add_argument("--labels", nargs="+", help="panel names, one per profile (default: the profile file names)")
    ap.add_argument("--lam", type=float, nargs="+", default=[0.02], help="lambda(s): one block of two rows per lambda (small = bigger file)")
    ap.add_argument("--oracle", action="store_true", help="exact partitioner (slow: builds cost tables, about 4 s per tile per profile)")
    ap.add_argument("--policy", help="trained .ambipolicy used for every profile (instead of --oracle)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--crop", type=int, default=256, help="crop side in px (multiple of 64) at a random position; 0 = whole pictures")
    ap.add_argument("--crops-per-image", type=int, default=1)
    ap.add_argument("--max-images", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jpeg", choices=["size", "psnr", "none"], default="size",
                    help="JPEG row: size = same file size as the AMBI panel above it; psnr = same PSNR; none")
    ap.add_argument("--zoom", type=int, default=0, help="also show a magnified patch of this size (px) under each panel")
    ap.add_argument("--zoom-at", type=int, nargs=2, metavar=("X", "Y"), help="top-left corner of the zoom patch (default: centre)")
    ap.add_argument("--analysis-dir", help="results go to ANALYSIS\\<name>_profiles and ANALYSIS\\<name>_profiles_files "
                                           "(default: <the picture folder>_analysis)")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (needs --oracle), e.g. 16; default 1")
    ap.add_argument("--json", help="also write every number to this file")
    a = ap.parse_args()
    if a.workers > 1 and not a.oracle:
        sys.exit("--workers needs --oracle (a policy run uses one process and the GPU)")
    if len(a.profiles) < 2:
        sys.exit("give at least two profiles (one is just side_by_side.py)")
    if not a.oracle and not a.policy:
        sys.exit("give --oracle or --policy FILE")
    if a.labels and len(a.labels) != len(a.profiles):
        sys.exit("--labels needs one name per profile")
    profs = [Profile.load(p) for p in a.profiles]
    labels = a.labels or [pathlib.Path(p).stem for p in a.profiles]
    if len(set(labels)) < len(labels):
        sys.exit("two panels would get the same name; use --labels")
    part = None
    if a.policy and not a.oracle:
        from ambi_rl.partitioner import PolicyPartitioner
        part = PolicyPartitioner.load(a.policy, a.device)
    who = "exact oracle" if a.oracle else "policy " + pathlib.Path(a.policy).name

    if a.image:
        files = [pathlib.Path(a.image)]
        src_dir = files[0].resolve().parent
    else:
        src_dir = pathlib.Path(a.image_dir)
        files = [f for f in sorted(src_dir.rglob("*")) if f.suffix.lower() in EXTS]
        if not files:
            sys.exit(f"no images in {a.image_dir}")
        if a.max_images:
            files = files[:a.max_images]
    base = pathlib.Path(a.analysis_dir) if a.analysis_dir else src_dir.with_name(src_dir.name + "_analysis")
    prefix = base.name[:-len("_analysis")] if base.name.endswith("_analysis") else base.name
    cmp_dir, conv = base / f"{prefix}_profiles", base / f"{prefix}_profiles_files"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    conv.mkdir(parents=True, exist_ok=True)

    jobs = pick_crops(files, a.crop, a.crops_per_image, a.seed)
    if not jobs:
        sys.exit("nothing to draw (no picture is large enough for the crop)")
    lams = list(a.lam)
    live, t_start, records, secs = LiveLine(), time.time(), [], []

    pre = {}
    if a.workers > 1:
        tasks = []
        for k, (f, cy, cx, rgb, load_note, crop_at) in enumerate(jobs):
            for pi, (pr, lb, path) in enumerate(zip(profs, labels, a.profiles)):
                nt, wt, we = work_units(rgb, pr, True, len(lams))
                tasks.append((nt * wt + len(lams) * we, (k, pi, rgb, path, lb, lams, conv, stem_of(f, crop_at))))
        tasks.sort(key=lambda t: -t[0])
        running = min(a.workers, len(tasks))
        print(f"{len(jobs)} crops x {len(profs)} profiles = {len(tasks)} jobs on {running} worker processes", flush=True)
        import multiprocessing as mp
        q = mp.get_context("spawn").Queue()
        wcrop = {}
        for w_, (k, *_rest) in tasks:
            wcrop[k] = wcrop.get(k, 0.0) + w_
        prog, bar, n_done = {}, Bar(live, "all crops", sum(w for w, _ in tasks)), 0
        for i, res in run_tasks(encode_task, [t for _, t in tasks], a.workers, tick=0.5, initializer=init_worker, initargs=(q,)):
            if res is not None:
                k, pi, e, _ = res
                pre.setdefault(k, {})[pi] = e
                prog[(k, pi)] = tasks[i][0]
                n_done += 1
            drain(q, prog)
            bar.done = sum(prog.values())
            per = " ".join(f"{k + 1}:{min(100, 100 * sum(v for (kk, _), v in prog.items() if kk == k) / wcrop[k]):3.0f}%" for k in range(len(jobs)))
            bar.render(f"{n_done}/{len(tasks)} jobs | crops {per}")
        live.clear()
        print(f"all encodes done in {fmt_time(time.time() - t_start)}; drawing the sheets", flush=True)

    for k, (f, cy, cx, rgb, load_note, crop_at) in enumerate(jobs, 1):
        t_pic = time.time()
        h, w = rgb.shape[:2]
        px = h * w
        stem = stem_of(f, crop_at)
        on_disk = f.suffix.lower() == ".png" and crop_at is None and not load_note
        png_b = f.stat().st_size if on_disk else png_size(rgb)
        if pre:
            enc = [pre[k - 1][pi] for pi in range(len(profs))]
        else:
            total = sum(nt * wt + len(lams) * we for nt, wt, we in (work_units(rgb, pr, a.oracle, len(lams)) for pr in profs))
            nm = f.name if len(f.name) <= 16 else f.name[:8] + "..." + f.suffix
            bar = Bar(live, f"pic {k}/{len(jobs)} {nm}", total)
            enc = [encode_profile(rgb, pr, lb, lams, a.oracle, part, bar, conv, stem) for pr, lb in zip(profs, labels)]
            live.clear()
        if crop_at:
            Image.fromarray(rgb).save(conv / f"{stem}_original.png")

        png_panel = lambda: dict(
            head="PNG  (lossless)", head_color=HEAD_PNG, rgb=rgb,
            lines=[(f"size: {kb(png_b)}", True), (f"{png_b * 8 / px:.2f} bits per pixel", False),
                   ("PSNR: exact, no pixel changed", False), ("SSIM: 1.0000", False),
                   ("the original file" if on_disk else "PNG of the crop" if crop_at else "PNG of the pixels shown", False)])
        panels = []
        print(f"  {f.name}" + (f"  crop at x={crop_at[0]}, y={crop_at[1]}" if crop_at else "") + f"   PNG {png_b:,} B", flush=True)
        for li, lam in enumerate(lams):
            ambi_row, jpeg_row = [png_panel()], []
            legend = {"size": "JPEG row: each JPEG has the same file size as the AMBI panel above it (same bytes, compare quality).",
                      "psnr": "JPEG row: each JPEG has the same PSNR as the AMBI panel above it (same quality, compare bytes).",
                      "none": ""}[a.jpeg]
            jpeg_row.append(dict(note=True, lines=[(f"lambda {lam:g}", True), (legend, False)] if legend else [(f"lambda {lam:g}", True)]))
            print(f"    lambda {lam:g}", flush=True)
            ref = None
            for pi, (pr, lb) in enumerate(zip(profs, labels)):
                e = enc[pi][li]
                nb, ps, ss = e["nb"], e["psnr"], e["ssim"]
                ref = ref or (nb, ps, ss)
                lines = [(f"size: {kb(nb)}", True), (f"{nb * 8 / px:.2f} bits per pixel", False),
                         (f"PSNR: {ps:.2f} dB      SSIM: {ss:.4f}", True),
                         (f"{100 * nb / png_b:.1f}% of the PNG  ({png_b / nb:.1f}x smaller)", True),
                         (f"profile id {pr.profile_id}, " + ("intra: " + " ".join(pr.cfg.intra_preds) if pr.cfg.intra_preds else "no intra"), False)]
                if pi:
                    lines.append((f"vs {labels[0]}: size {100 * (nb / ref[0] - 1):+.1f}%, {ps - ref[1]:+.2f} dB, {ss - ref[2]:+.4f} SSIM", True))
                lines.append((f"encode {e['t_enc']:.1f} s, decode {e['t_dec']:.1f} s, deblock level {e['deblock']}", False))
                ambi_row.append(dict(head=f"{lb}  (lambda {lam:g})", head_color=PROFILE_COLORS[pi % len(PROFILE_COLORS)], rgb=e["dec"], lines=lines))
                rec = dict(image=f.name, y=cy, x=cx, lam=lam, profile=lb, profile_id=pr.profile_id, png_bytes=png_b, bytes=nb, psnr=ps, ssim=ss)
                if a.jpeg != "none":
                    q, data, ok = jpeg_match_size(rgb, nb) if a.jpeg == "size" else jpeg_match_psnr(rgb, ps)
                    (conv / f"{stem}_{safe(lb)}_lam{lam:g}.jpg").write_bytes(data)
                    dj = jpeg_decode(data)
                    pj, sj, nj = psnr_rgb(rgb, dj), ssim_y(rgb, dj), len(data)
                    jl = [(f"size: {kb(nj)}", True), (f"{nj * 8 / px:.2f} bits per pixel", False),
                          (f"PSNR: {pj:.2f} dB      SSIM: {sj:.4f}", True),
                          (f"{100 * nj / png_b:.1f}% of the PNG  ({png_b / nj:.1f}x smaller)", True)]
                    if a.jpeg == "size":
                        jl.append((f"same size as {lb}" if ok else "quality 1 is JPEG's floor: it cannot go this small", False))
                        if not ok:
                            jl.append((f"NOT equal size: this JPEG is {nj / nb:.1f}x the AMBI file", True))
                        jl.append((f"vs AMBI: {pj - ps:+.2f} dB, {sj - ss:+.4f} SSIM", True))
                    else:
                        jl.append((f"same PSNR as {lb}" if ok else "cannot reach AMBI's PSNR (quality 100)", False))
                        jl.append((f"vs AMBI: file {100 * (nj / nb - 1):+.0f}% (SSIM {sj - ss:+.4f})", True))
                    jpeg_row.append(dict(head=f"JPEG  (quality {q})", head_color=HEAD_JPEG, rgb=dj, lines=jl))
                    rec.update(jpeg_quality=q, jpeg_bytes=nj, jpeg_psnr=pj, jpeg_ssim=sj, jpeg_matched=bool(ok))
                records.append(rec)
                print(f"      {lb:<18} {nb:>9,} B  {100 * nb / png_b:5.1f}% of PNG ({png_b / nb:5.1f}x)  {ps:6.2f} dB  SSIM {ss:.4f}"
                      + (f"   | JPEG q{q} {nj:,} B {pj:.2f} dB SSIM {sj:.4f}" if a.jpeg != "none" else ""), flush=True)
            panels += ambi_row + (jpeg_row if a.jpeg != "none" else [])
        cols = len(profs) + 1
        zs = min(a.zoom, h, w) if a.zoom else 0
        zpos = None
        if zs:
            zx, zy = a.zoom_at if a.zoom_at else ((w - zs) // 2, (h - zs) // 2)
            zpos = (int(np.clip(zx, 0, w - zs)), int(np.clip(zy, 0, h - zs)))
        title = [f"{f.name}   {w}x{h} px" + (f"   (crop at x={crop_at[0]}, y={crop_at[1]})" if crop_at else ""),
                 f"AMBI profiles, left to right: {' | '.join(labels)}     partitioner: {who}",
                 "Same lambda for every profile, so sizes and qualities differ a little between profiles; "
                 + ("each JPEG is matched to the AMBI panel above it." if a.jpeg != "none" else "no JPEG row.")]
        canvas = compose(panels, title, bool(zs), zs, zpos, cols)
        name = cmp_dir / f"{stem}_lam{'-'.join(f'{l:g}' for l in lams)}_profiles.png"
        canvas.save(name)
        secs.append(time.time() - t_pic)
        left = (len(jobs) - k) * float(np.mean(secs))
        print(f"  wrote {name}  ({canvas.size[0]}x{canvas.size[1]})  picture {fmt_time(secs[-1])}"
              + (f", about {fmt_time(left)} left" if left > 0 else ""), flush=True)

    print(f"\nPooled over {len(jobs)} picture(s); compression = total AMBI bytes / total PNG bytes, quality = mean over the pictures")
    for lam in lams:
        print(f"\nlambda {lam:g}\n  {'profile':<18} {'% of PNG':>9} {'x smaller':>10} {'PSNR dB':>8} {'SSIM(Y)':>8}"
              + (f"   {'JPEG same ' + a.jpeg:>20}: {'PSNR dB':>8} {'SSIM(Y)':>8}   AMBI minus JPEG" if a.jpeg != "none" else ""))
        for lb in labels:
            rs = [r for r in records if r["lam"] == lam and r["profile"] == lb]
            tb, tp = sum(r["bytes"] for r in rs), sum(r["png_bytes"] for r in rs)
            ps, ss = np.mean([r["psnr"] for r in rs]), np.mean([r["ssim"] for r in rs])
            row = f"  {lb:<18} {100 * tb / tp:8.2f}% {tp / tb:9.1f}x {ps:8.2f} {ss:8.4f}"
            if a.jpeg != "none":
                pj, sj = np.mean([r["jpeg_psnr"] for r in rs]), np.mean([r["jpeg_ssim"] for r in rs])
                tj = sum(r["jpeg_bytes"] for r in rs)
                row += (f"   {'':>20}  {pj:8.2f} {sj:8.4f}   {ps - pj:+.2f} dB, {ss - sj:+.4f} SSIM" if a.jpeg == "size"
                        else f"   {'':>20}  {pj:8.2f} {sj:8.4f}   JPEG file {100 * (tj / tb - 1):+.0f}% bigger")
            print(row)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(records, indent=1))
        print(f"\nwrote {a.json}")
    print(f"\nsheets:          {cmp_dir}\nconverted files: {conv}\ntotal time {fmt_time(time.time() - t_start)}")


if __name__ == "__main__":
    main()
