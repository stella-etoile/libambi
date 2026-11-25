import argparse, csv, os, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ambi_core import psnr_rgb
from ambi_core.bitstream import decode_image, encode_image
from ambi_core.metrics import ssim_y
from ambi_core.profile import Profile
from ambi_cli.main import load_rgb

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
FONTS = ("consola.ttf", "DejaVuSansMono.ttf", "cour.ttf", "LiberationMono-Regular.ttf", "arial.ttf")
BG, FG, DIM, ACC = (24, 24, 26), (232, 232, 232), (150, 150, 155), (120, 190, 255)
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
NEAREST = getattr(Image, "Resampling", Image).NEAREST


def human(n) -> str:
    return f"{int(n):,}"


def rel(nbytes: float, ref: float) -> str:
    d = 100 * (1 - nbytes / ref)
    return f"{abs(d):.1f}% smaller" if d >= 0 else f"{abs(d):.1f}% LARGER"


def db(x: float) -> str:
    return "identical" if not np.isfinite(x) else f"{x:.2f} dB"


def pick_device(spec: str) -> str:
    if spec != "auto":
        return spec
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def list_inputs(path: pathlib.Path):
    if path.is_dir():
        return [f for f in sorted(path.iterdir()) if f.is_file() and f.suffix.lower() in EXTS]
    return [path]


def get_font(size: int):
    for name in FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def panel_lines(g: int, t: dict, orig: dict) -> list:
    """(text, colour) lines under one panel (kept under 52 characters). g = 0 is the original."""
    if g == 0:
        return [("ORIGINAL", ACC),
                (f"format       {orig['fmt']}", FG),
                (f"file         {human(orig['bytes'])} bytes", FG),
                (f"raw RGB      {human(orig['raw'])} bytes", FG),
                (f"size         {orig['w']} x {orig['h']} px", FG),
                (f"bitrate      {orig['bytes'] * 8 / (orig['w'] * orig['h']):.3f} bpp (as a file)", FG),
                ("the reference for PSNR and SSIM", DIM)]
    n = "1 CYCLE" if g == 1 else f"{g} CYCLES"
    L = [(f"AFTER {n} OF ENCODE + DECODE", ACC),
         (f".ambi file   {human(t['ambi_bytes'])} bytes (encode #{g})", FG),
         (f"vs original  {rel(t['ambi_bytes'], orig['bytes'])} ({orig['bytes'] / t['ambi_bytes']:.1f}:1)", FG),
         (f"vs raw RGB   {rel(t['ambi_bytes'], orig['raw'])}", FG),
         (f"bitrate      {t['ambi_bytes'] * 8 / (orig['w'] * orig['h']):.3f} bpp", FG),
         (f"PSNR         {db(t['psnr_o'])}  (vs original)", FG),
         (f"SSIM(Y)      {t['ssim_o']:.4f}  (vs original)", FG),
         (f"last cycle alone: PSNR {db(t['psnr_p'])}", DIM),
         (f"                  SSIM {t['ssim_p']:.4f}", DIM)]
    if g > 1:
        L.append((f"file vs 1st encode  {t['vs_first']:+.1f}%", DIM))
    pr = t.get("prune")
    if pr:
        L += [(f"pruning       k={pr['k']}: looked at {pr['looked']:.1f}% of rects", DIM),
              (f"              priced {pr['priced']:.1f}% as leaves, {pr['ms']:.0f} ms/tile", DIM)]
    else:
        L.append((f"pruning       none ({t['how']})", DIM))
    L += [(f"deblock level {t['deblock']}", DIM),
          (f"this cycle    encode {t['enc_s']:.1f} s, decode {t['dec_s']:.2f} s", DIM),
          (f"total time    {t['cum_s']:.1f} s for {g} cycle{'s' if g > 1 else ''}", DIM),
          (f"decoded PNG   {human(t['png_bytes'])} bytes (lossless)", DIM)]
    return L


def make_composite(name, how, orig, shown, pics, tele, a, out_path):
    """orig: dict; shown: list of generation numbers; pics: {0: original array, g: decoded array}."""
    H, W = orig["h"], orig["w"]
    cols = [0] + shown
    iw = min(W, a.panel_width)
    scale = iw / W
    ih = max(1, round(H * scale))
    pw = max(iw, 520)
    f_big, f_txt = get_font(22), get_font(15)
    line_h = int(15 * 1.45)
    n_lines = max(len(panel_lines(g, tele.get(g), orig)) for g in cols)
    pad = 16
    zoom_row = scale < 1.0
    c = max(8, min(min(W, H), pw // a.zoom)) if zoom_row else 0
    row2_h = c * a.zoom if zoom_row else 0
    head_h = 70
    cap_h = 22
    total_w = pad + len(cols) * (pw + pad)
    total_h = head_h + ih + pad + (cap_h + row2_h + pad if zoom_row else 0) + n_lines * line_h + pad * 2
    img = Image.new("RGB", (total_w, total_h), BG)
    d = ImageDraw.Draw(img)
    d.text((pad, 12), name, font=f_big, fill=FG)
    d.text((pad, 42), how, font=f_txt, fill=DIM)
    for i, g in enumerate(cols):
        x0 = pad + i * (pw + pad)
        pic = Image.fromarray(pics[g])
        whole = pic.resize((iw, ih), LANCZOS) if scale < 1.0 else pic
        img.paste(whole, (x0 + (pw - iw) // 2, head_h))
        y = head_h + ih + pad
        if zoom_row:
            cy, cx = (H - c) // 2, (W - c) // 2
            crop = pic.crop((cx, cy, cx + c, cy + c)).resize((c * a.zoom, c * a.zoom), NEAREST)
            if i == 0:
                d.text((pad, y - 2), f"top: whole picture scaled to {100 * scale:.0f}%.   below it: the centre {c}x{c} px at native "
                       f"pixels, zoomed {a.zoom}x", font=f_txt, fill=DIM)
            img.paste(crop, (x0 + (pw - c * a.zoom) // 2, y + cap_h))
            y += cap_h + row2_h + pad
        for text, col in panel_lines(g, tele.get(g), orig):
            d.text((x0, y), text, font=f_txt, fill=col)
            y += line_h
    img.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="encode + decode again and again; original, 1x, 5x, 10x side by side with all the numbers")
    ap.add_argument("input", help="an image file or a folder of images")
    ap.add_argument("-o", "--out-dir", default="roundtrip_out")
    ap.add_argument("--gens", type=int, nargs="+", default=[1, 5, 10], metavar="N",
                    help="which numbers of encode/decode cycles to show and save (default 1 5 10)")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.03, help="rate/quality trade-off (higher = smaller file), default 0.03")
    ap.add_argument("--profile", default=os.path.join("profiles", "clic-base-2k.ambiprofile"))
    ap.add_argument("--policy", default=os.path.join("policies", "clic2k_big.ambipolicy"), help="trained .ambipolicy (the partitioner)")
    ap.add_argument("--oracle", action="store_true", help="use the exact oracle instead of the policy (slow)")
    ap.add_argument("--prune-k", type=int, default=5, metavar="K",
                    help="exact search over the policy's K best actions per rectangle (default 5; 0 = the policy alone)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), metavar="N",
                    help="processes that price the candidate leaves (default: half the logical cores = the physical cores)")
    ap.add_argument("--device", default="auto", help="where the policy network runs: auto, cpu, cuda (default auto)")
    ap.add_argument("--mu", type=float, default=0.0, help="extra per-block bit cost (bias toward larger blocks)")
    ap.add_argument("--panel-width", type=int, default=640, help="width of each panel in the side-by-side picture (px); bigger pictures are scaled down")
    ap.add_argument("--zoom", type=int, default=3, help="zoom of the native-pixel centre crops shown under scaled-down panels (default 3)")
    a = ap.parse_args()

    gens = sorted(set(a.gens))
    if gens[0] < 1:
        sys.exit("--gens takes cycle counts of 1 or more")
    last = gens[-1]
    files = list_inputs(pathlib.Path(a.input))
    if not files:
        sys.exit(f"no images found in {a.input}")
    if not pathlib.Path(a.profile).exists():
        sys.exit(f"profile not found: {a.profile}  (give --profile FILE)")
    prof = Profile.load(a.profile)

    part, how, how_short = None, "exact oracle", "exact oracle"
    if not a.oracle:
        if not pathlib.Path(a.policy).exists():
            sys.exit(f"policy not found: {a.policy}  (give --policy FILE, or --oracle for the slow exact partitioner)")
        from ambi_rl.pruned import make_partitioner
        device = pick_device(a.device)
        workers = a.workers if a.prune_k else 1
        part = make_partitioner(a.policy, prof, device, a.prune_k, workers)
        how = (f"policy {pathlib.Path(a.policy).name} + exact search over its top {a.prune_k} actions, {workers} worker(s), {device}"
               if a.prune_k else f"policy {pathlib.Path(a.policy).name} alone, {device}")
        how_short = "policy alone"
    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    banner = f"profile {prof.name} (id {prof.profile_id}), lambda {a.lam:g}, partitioner: {how}"
    print(banner)
    print(f"output folder: {out_dir.resolve()}   cycles shown: {', '.join(map(str, gens))}   (runs {last} cycle(s) per image)\n")

    csv_path = out_dir / "generations.csv"
    cols = ["image", "width", "height", "original_bytes", "raw_bytes", "cycle", "ambi_bytes", "pct_smaller_than_original_file",
            "pct_smaller_than_raw", "bpp", "psnr_vs_original_db", "ssim_y_vs_original", "psnr_vs_previous_db", "ssim_y_vs_previous",
            "deblock_level", "encode_s", "decode_s", "files_saved", "prune_k", "pct_rects_looked_at", "pct_rects_priced_as_leaf",
            "partition_ms_per_tile"]
    used, summary = set(), {g: [] for g in gens}
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        cw = csv.writer(cf)
        cw.writerow(cols)
        try:
            for i, f in enumerate(files, 1):
                rgb, had_alpha = load_rgb(f)
                H, W = rgb.shape[:2]
                orig = dict(fmt=f.suffix.lstrip(".").upper(), bytes=f.stat().st_size, raw=H * W * 3, w=W, h=H)
                stem, n = f.stem, 1
                while stem in used:
                    n += 1
                    stem = f"{f.stem}-{n}"
                used.add(stem)
                print(f"[{i}/{len(files)}] {f.name}   {W}x{H}   {human(orig['bytes'])} bytes on disk, {human(orig['raw'])} raw"
                      + ("   (alpha channel dropped)" if had_alpha else ""))

                prev, tele, pics, first_bytes, cum = rgb, {}, {0: rgb}, None, 0.0
                for g in range(1, last + 1):
                    if hasattr(part, "reset_stats"):
                        part.reset_stats()
                    t0 = time.time()
                    res = encode_image(prev, prof, a.lam, a.mu, partitioner=part)
                    enc_s = time.time() - t0
                    pr = None
                    if hasattr(part, "stats") and hasattr(part, "k"):
                        st, nt, nr = part.stats, max(1, part.stats["tiles"]), part.space.n_rects
                        pr = dict(k=part.k, looked=100 * st["looked_at"] / nt / nr, priced=100 * st["priced_as_leaf"] / nt / nr,
                                  ms=1000 * st["seconds_wall"] / nt)
                    data, saved = res.data, g in gens
                    if saved:
                        p = out_dir / f"{stem}.gen{g:02d}.ambi"
                        p.write_bytes(data)
                        data = p.read_bytes()
                    t0 = time.time()
                    dec = decode_image(data, prof)
                    dec_s = time.time() - t0
                    cum += enc_s + dec_s
                    first_bytes = first_bytes or len(data)
                    t = dict(ambi_bytes=len(data), enc_s=enc_s, dec_s=dec_s, cum_s=cum, deblock=res.deblock,
                             psnr_o=psnr_rgb(rgb, dec), ssim_o=ssim_y(rgb, dec), psnr_p=psnr_rgb(prev, dec), ssim_p=ssim_y(prev, dec),
                             vs_first=100 * (len(data) / first_bytes - 1), prune=pr, how=how_short)
                    if saved:
                        q = out_dir / f"{stem}.gen{g:02d}.decoded.png"
                        Image.fromarray(dec).save(q)
                        t["png_bytes"] = q.stat().st_size
                        pics[g] = dec
                        summary[g].append((orig["bytes"], orig["raw"], len(data), H * W, t["psnr_o"], t["ssim_o"]))
                    tele[g] = t
                    cw.writerow([f.name, W, H, orig["bytes"], orig["raw"], g, len(data), f"{100 * (1 - len(data) / orig['bytes']):.2f}",
                                 f"{100 * (1 - len(data) / orig['raw']):.2f}", f"{len(data) * 8 / (H * W):.4f}", f"{t['psnr_o']:.3f}",
                                 f"{t['ssim_o']:.5f}", f"{t['psnr_p']:.3f}", f"{t['ssim_p']:.5f}", res.deblock,
                                 f"{enc_s:.2f}", f"{dec_s:.2f}", "yes" if saved else "no",
                                 pr["k"] if pr else 0, f"{pr['looked']:.2f}" if pr else "", f"{pr['priced']:.2f}" if pr else "",
                                 f"{pr['ms']:.1f}" if pr else ""])
                    cf.flush()
                    print(f"  cycle {g:>2}: {human(len(data)):>10} bytes ({rel(len(data), orig['bytes'])} than the file, "
                          f"{len(data) * 8 / (H * W):.3f} bpp)   PSNR {db(t['psnr_o'])}  SSIM(Y) {t['ssim_o']:.4f}  vs original"
                          f"   | this cycle alone: {db(t['psnr_p'])}   [{enc_s:.1f} s + {dec_s:.1f} s]"
                          + (f"   | pruning k={pr['k']}: looked at {pr['looked']:.1f}% of rectangles, priced {pr['priced']:.1f}% as leaves"
                             f" ({pr['ms']:.0f} ms/tile)" if pr else "   | no pruning ({})".format(how_short))
                          + ("   saved" if saved else ""), flush=True)
                    prev = dec

                comp = out_dir / f"{stem}.generations.png"
                make_composite(f.name, banner, orig, gens, pics, tele, a, comp)
                print(f"  -> {comp.name}\n", flush=True)
        finally:
            if hasattr(part, "close"):
                part.close()

    if len(summary[gens[0]]) > 1:
        print("totals over all images")
        for g in gens:
            rows = summary[g]
            orig_b, raw_b, amb, pix = (sum(r[k] for r in rows) for k in (0, 1, 2, 3))
            print(f"  after {g:>2} cycle(s): {human(orig_b)} bytes originals -> {human(amb)} bytes .ambi, {rel(amb, orig_b)} than the originals, "
                  f"{rel(amb, raw_b)} than raw, {amb * 8 / pix:.3f} bpp;  mean PSNR {np.mean([r[4] for r in rows]):.2f} dB, "
                  f"SSIM(Y) {np.mean([r[5] for r in rows]):.4f}")
    print(f"\ncycle-by-cycle numbers for every image: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())