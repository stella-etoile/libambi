import argparse, io, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ambi_core import psnr_rgb
from ambi_core.bitstream import decode_image, encode_image
from ambi_core.metrics import ssim_y
from ambi_core.profile import Profile

EXTS = (".png", ".webp", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
BG, INK, MUTED = (255, 255, 255), (20, 20, 20), (90, 90, 90)
MIN_PANEL_W = 440
HEAD_PNG, HEAD_AMBI, HEAD_JPEG = (70, 70, 70), (30, 90, 170), (176, 82, 20)


def load_font(size):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def fit_font(lines, width, start, lo=10):
    for size in range(start, lo - 1, -1):
        f = load_font(size)
        if all(f.getlength(t) <= width - 16 for t, _ in lines):
            return f, size
    return load_font(lo), lo


def wrap_text(text, font, width):
    lines, cur = [], ""
    for word in text.split():
        trial = word if not cur else cur + " " + word
        if cur and font.getlength(trial) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def load_rgb(path):
    im = ImageOps.exif_transpose(Image.open(path))
    note = None
    if im.mode == "F":
        arr = np.asarray(im, dtype=np.float64)
        arr = arr * 255.0 if arr.max() <= 1.0 else arr
        im, note = Image.fromarray(np.clip(np.nan_to_num(arr).round(), 0, 255).astype(np.uint8), "L"), "float image scaled to 8-bit"
    if im.mode in ("I;16", "I;16L", "I;16B", "I"):
        arr = np.asarray(im, dtype=np.float64)
        if arr.max() > 255:
            arr = arr / 65535.0 * 255.0
        im, note = Image.fromarray(np.clip(arr.round(), 0, 255).astype(np.uint8), "L"), "16-bit scaled to 8-bit"
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        if rgba.getchannel("A").getextrema()[0] < 255:
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            bg.alpha_composite(rgba)
            im, note = bg, "transparency flattened onto white"
    return np.array(im.convert("RGB")), note


def kb(n):
    return f"{n / 1024:,.1f} KB ({n:,} bytes)"


def png_size(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG", optimize=True, compress_level=9)
    return len(buf.getvalue())


def jpeg_bytes(rgb, q):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=int(q), subsampling="4:2:0", optimize=True)
    return buf.getvalue()


def jpeg_decode(data):
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def jpeg_match_size(rgb, target_bytes):
    lo, hi, best = 1, 100, None
    while lo <= hi:
        mid = (lo + hi) // 2
        data = jpeg_bytes(rgb, mid)
        if len(data) <= target_bytes:
            best, lo = (mid, data), mid + 1
        else:
            hi = mid - 1
    return (best[0], best[1], True) if best else (1, jpeg_bytes(rgb, 1), False)


def jpeg_match_psnr(rgb, target_db):
    lo, hi, best = 1, 100, None
    while lo <= hi:
        mid = (lo + hi) // 2
        data = jpeg_bytes(rgb, mid)
        if psnr_rgb(rgb, jpeg_decode(data)) >= target_db:
            best, hi = (mid, data), mid - 1
        else:
            lo = mid + 1
    return (best[0], best[1], True) if best else (100, jpeg_bytes(rgb, 100), False)


def zoom_patch(rgb, zs, zx, zy, width):
    patch = Image.fromarray(np.ascontiguousarray(rgb[zy:zy + zs, zx:zx + zs]))
    k = max(1, width // zs)
    return patch.resize((zs * k, zs * k), Image.NEAREST), k


def compose(panels, title, zoom, zs, zpos, cols):
    w0, h0 = panels[0]["rgb"].shape[1], panels[0]["rgb"].shape[0]
    S = max(1, -(-MIN_PANEL_W // w0))          # integer enlargement so small pictures get a readable panel
    pw, ph = w0 * S, h0 * S
    gap, margin = 16, 16
    all_lines = [ln for p in panels if not p.get("note") for ln in p["lines"]]     # a note panel (text only) does not size the font
    font, fs = fit_font(all_lines, pw, start=max(14, pw // 36))
    big = load_font(fs + 3)
    line_h = fs + 7
    head_h = fs + 16
    zk = max(1, pw // zs) if zoom else 1
    zoom_h = (8 + zs * zk + line_h) if zoom else 0
    title = list(title) if isinstance(title, (list, tuple)) else [title]
    if S > 1:
        title.append(f"pictures enlarged {S}x for display (nearest neighbour); sizes, PSNR and SSIM are for the real pixels")
    title_h = len(title) * (line_h + 2) + 12
    rows = [panels[i:i + cols] for i in range(0, len(panels), cols)]
    ncols = min(cols, len(panels))
    row_h = [head_h + ph + zoom_h + 8 + max(len(p["lines"]) for p in r) * line_h + 10 for r in rows]
    W = margin * 2 + ncols * pw + (ncols - 1) * gap
    H = margin + title_h + sum(row_h) + gap * (len(rows) - 1) + margin
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    for ti, t in enumerate(title):
        d.text((margin, margin + ti * (line_h + 2)), t, fill=INK if ti == 0 else MUTED, font=big if ti == 0 else font)
    ry = margin + title_h
    for r, rh in zip(rows, row_h):
        for i, p in enumerate(r):
            x0 = margin + i * (pw + gap)
            y = ry
            if p.get("note"):
                y += 8
                for text, strong in p["lines"]:
                    for part in wrap_text(text, font, pw - 8):
                        d.text((x0 + 2, y), part, fill=INK if strong else MUTED, font=font)
                        y += line_h
                continue
            d.rectangle([x0, y, x0 + pw, y + head_h], fill=p["head_color"])
            d.text((x0 + 8, y + 6), p["head"], fill=(255, 255, 255), font=big)
            y += head_h
            img = Image.fromarray(p["rgb"])
            canvas.paste(img.resize((pw, ph), Image.NEAREST) if S > 1 else img, (x0, y))
            d.rectangle([x0, y, x0 + pw - 1, y + ph - 1], outline=(190, 190, 190))
            y += ph
            if zoom:
                patch, k = zoom_patch(p.get("zoom_rgb", p["rgb"]), zs, zpos[0], zpos[1], pw)
                y += 8
                canvas.paste(patch, (x0, y))
                y += patch.size[1]
                d.text((x0 + 2, y + 2), f"zoom x{k} at x={zpos[0]}, y={zpos[1]} ({zs}x{zs} px)", fill=MUTED, font=font)
                y += line_h
            y += 8
            for text, strong in p["lines"]:
                d.text((x0 + 2, y), text, fill=INK if strong else MUTED, font=font)
                y += line_h
        ry += rh + gap
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="one picture")
    src.add_argument("--image-dir", help="pick pictures from this folder (random, see --n/--seed)")
    ap.add_argument("--n", type=int, default=1, help="how many pictures to pick from --image-dir")
    ap.add_argument("--seed", type=int, default=0, help="random pick / random crop position (same seed = same pictures)")
    ap.add_argument("--lam", type=float, nargs="+", default=[0.03],
                    help="one or more lambdas (small = bigger file, better quality); each gets an AMBI panel")
    ap.add_argument("--profile", default="profiles/photo-demo.ambiprofile")
    ap.add_argument("--policy", help="trained .ambipolicy (fast partitioner)")
    ap.add_argument("--oracle", action="store_true", help="exact partitioner (slow, about 4 s per 64x64 tile per lambda)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--crop", type=int, default=0, help="crop side in px (multiple of 64) taken at a random position; 0 = whole picture")
    ap.add_argument("--zoom", type=int, default=0, help="also show a magnified patch of this size (px) under each panel")
    ap.add_argument("--zoom-at", type=int, nargs=2, metavar=("X", "Y"), help="top-left corner of the zoom patch (default: centre)")
    ap.add_argument("--jpeg", choices=["size", "psnr", "none"], default="size",
                    help="JPEG panel per lambda: size = same file size as the AMBI file; psnr = same PSNR as AMBI; none")
    ap.add_argument("--jpeg-quality", type=int, metavar="Q", help="one JPEG at this fixed quality (1-100) instead of matched ones")
    ap.add_argument("--cols", type=int, default=3, help="panels per row")
    ap.add_argument("--analysis-dir", help="results go to ANALYSIS\\<name>_compared and ANALYSIS\\<name>_converted "
                                           "(default: <the picture folder>_analysis, e.g. kodak -> kodak_analysis\\kodak_compared)")
    ap.add_argument("--compare-dir", help="override: folder for the side-by-side pictures")
    ap.add_argument("--convert-dir", help="override: folder for the converted files (.ambi, .jpg, decoded/cropped PNGs)")
    ap.add_argument("--save-decoded", action="store_true", help="also save the decoded AMBI pixels as a PNG")
    a = ap.parse_args()
    if not a.policy and not a.oracle:
        sys.exit("give --policy FILE or --oracle")

    prof = Profile.load(a.profile)
    part = None
    if a.policy and not a.oracle:
        from ambi_rl.partitioner import PolicyPartitioner
        part = PolicyPartitioner.load(a.policy, a.device)
    who = "exact oracle" if a.oracle else "policy " + pathlib.Path(a.policy).name
    rng = np.random.default_rng(a.seed)

    if a.image:
        files = [pathlib.Path(a.image)]
    else:
        pool = sorted(f for f in pathlib.Path(a.image_dir).rglob("*") if f.suffix.lower() in EXTS)
        if not pool:
            sys.exit(f"no images in {a.image_dir}")
        files = [pool[i] for i in sorted(rng.choice(len(pool), size=min(a.n, len(pool)), replace=False))]
    if a.analysis_dir:
        base = pathlib.Path(a.analysis_dir)
    else:
        src_dir = pathlib.Path(a.image_dir) if a.image_dir else pathlib.Path(a.image).resolve().parent
        base = src_dir.with_name(src_dir.name + "_analysis") if src_dir.name else pathlib.Path("out/analysis")
    prefix = base.name[:-len("_analysis")] if base.name.endswith("_analysis") else base.name
    cmp_dir = pathlib.Path(a.compare_dir) if a.compare_dir else base / f"{prefix}_compared"
    conv = pathlib.Path(a.convert_dir) if a.convert_dir else base / f"{prefix}_converted"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    conv.mkdir(parents=True, exist_ok=True)

    failed = []
    for f in files:
        try:
            full, load_note = load_rgb(f)
            H, W = full.shape[:2]
            crop_at = None
            rgb = full
            if a.crop:
                if H < a.crop or W < a.crop:
                    print(f"skip {f.name}: smaller than the crop ({W}x{H})")
                    continue
                cy = int(rng.integers(0, (H - a.crop) // 64 + 1)) * 64
                cx = int(rng.integers(0, (W - a.crop) // 64 + 1)) * 64
                rgb = np.ascontiguousarray(full[cy:cy + a.crop, cx:cx + a.crop])
                crop_at = (cx, cy)
            h, w = rgb.shape[:2]
            px = h * w
            stem = f"{f.stem[:20]}_{f.suffix.lower().lstrip('.')}" + (f"_crop{crop_at[0]}_{crop_at[1]}" if crop_at else "")

            on_disk = f.suffix.lower() == ".png" and crop_at is None and not load_note
            png_b = f.stat().st_size if on_disk else png_size(rgb)
            if on_disk:
                png_note = "the original file"
            elif crop_at:
                png_note = "PNG of this crop" + (f" (from {f.suffix.lower()})" if f.suffix.lower() != ".png" else "")
            elif f.suffix.lower() == ".png":
                png_note = "PNG of the pixels shown"
            else:
                png_note = f"PNG made from the decoded {f.suffix.lower()}"
            if load_note:
                png_note += f"; {load_note}"
            panels = [dict(
                head="PNG  (lossless)", head_color=HEAD_PNG, rgb=rgb,
                lines=[(f"size: {kb(png_b)}", True), (f"{png_b * 8 / px:.2f} bits per pixel", False),
                       ("PSNR: exact, no pixel changed", False), ("SSIM: 1.0000", False), (png_note, False)])]

            def jpeg_panel(q, data, notes):
                dec_j = jpeg_decode(data)
                ps_j, ss_j = psnr_rgb(rgb, dec_j), ssim_y(rgb, dec_j)
                nj = len(data)
                return ps_j, ss_j, nj, dict(
                    head=f"JPEG  (lossy, quality {q})", head_color=HEAD_JPEG, rgb=dec_j,
                    lines=[(f"size: {kb(nj)}", True), (f"{nj * 8 / px:.2f} bits per pixel", False),
                           (f"PSNR: {ps_j:.2f} dB      SSIM: {ss_j:.4f}", True),
                           (f"{100 * nj / png_b:.1f}% of the PNG  ({png_b / nj:.1f}x smaller)", True)] + notes)

            if a.jpeg_quality:
                q = int(np.clip(a.jpeg_quality, 1, 100))
                data = jpeg_bytes(rgb, q)
                (conv / f"{stem}_q{q}.jpg").write_bytes(data)
                ps_j, ss_j, nj, pan = jpeg_panel(q, data, [("fixed quality setting, 4:2:0 chroma", False)])
                panels.append(pan)
                print(f"  {f.name}  JPEG q{q}: {nj:,} B ({png_b / nj:.1f}x smaller)  {ps_j:.2f} dB  SSIM {ss_j:.4f}", flush=True)

            tabs = None
            if a.oracle:
                from ambi_core.search import TileTables
                tabs = TileTables(rgb, prof)
            for lam in a.lam:
                t0 = time.time()
                r = encode_image(rgb, prof, lam, tables=tabs.tables) if a.oracle else encode_image(rgb, prof, lam, partitioner=part)
                t_enc = time.time() - t0
                ambi_path = conv / f"{stem}_lam{lam:g}.ambi"
                ambi_path.write_bytes(r.data)
                t0 = time.time()
                dec = decode_image(ambi_path.read_bytes(), prof)
                t_dec = time.time() - t0
                ps, ss = psnr_rgb(rgb, dec), ssim_y(rgb, dec)
                if a.save_decoded:
                    Image.fromarray(dec).save(conv / f"{stem}_lam{lam:g}_decoded.png")
                nb = len(r.data)
                panels.append(dict(
                    head=f"AMBI  (lossy, lambda {lam:g})", head_color=HEAD_AMBI, rgb=dec,
                    lines=[(f"size: {kb(nb)}", True), (f"{nb * 8 / px:.2f} bits per pixel", False),
                           (f"PSNR: {ps:.2f} dB      SSIM: {ss:.4f}", True),
                           (f"{100 * nb / png_b:.1f}% of the PNG  ({png_b / nb:.1f}x smaller)", True),
                           (f"encode {t_enc:.1f} s, decode {t_dec:.1f} s, deblock level {r.deblock}", False)]))
                if a.jpeg != "none" and not a.jpeg_quality:
                    if a.jpeg == "size":
                        q, data, ok = jpeg_match_size(rgb, nb)
                    else:
                        q, data, ok = jpeg_match_psnr(rgb, ps)
                    (conv / f"{stem}_lam{lam:g}.jpg").write_bytes(data)
                    if a.jpeg == "size":
                        notes = [(f"same size as AMBI lambda {lam:g}" if ok else "quality 1 is JPEG's floor: it cannot go this small", False)]
                    else:
                        notes = [(f"same PSNR as AMBI lambda {lam:g}" if ok else "cannot reach AMBI's PSNR (quality 100)", False)]
                    ps_j, ss_j, nj, pan = jpeg_panel(q, data, notes)
                    if a.jpeg == "size" and not ok:
                        pan["lines"].append((f"NOT equal size: this JPEG is {nj / nb:.1f}x the AMBI file", True))
                        cmp = f"vs AMBI: {ps_j - ps:+.2f} dB, {ss_j - ss:+.4f} SSIM"
                    elif a.jpeg == "size":
                        cmp = f"vs AMBI: {ps_j - ps:+.2f} dB, {ss_j - ss:+.4f} SSIM"
                    else:
                        cmp = f"vs AMBI: file {100 * (nj / nb - 1):+.0f}% (SSIM {ss_j - ss:+.4f})"
                    pan["lines"].append((cmp, True))
                    panels.append(pan)
                print(f"  {f.name}  lambda {lam:g}: PNG {png_b:,} B  AMBI {nb:,} B ({png_b / nb:.1f}x smaller)  {ps:.2f} dB  SSIM {ss:.4f}", flush=True)
                if a.jpeg != "none" and not a.jpeg_quality:
                    print(f"      JPEG q{q}: {nj:,} B ({png_b / nj:.1f}x smaller)  {ps_j:.2f} dB  SSIM {ss_j:.4f}   [{'matched ' + a.jpeg}{'' if ok else ', NOT reached'}]", flush=True)

            zs = min(a.zoom, h, w) if a.zoom else 0
            zpos = None
            if zs:
                zx, zy = a.zoom_at if a.zoom_at else ((w - zs) // 2, (h - zs) // 2)
                zpos = (int(np.clip(zx, 0, w - zs)), int(np.clip(zy, 0, h - zs)))
            title = [f"{f.name}   {w}x{h} px" + (f"   (crop at x={crop_at[0]}, y={crop_at[1]} of {W}x{H})" if crop_at else ""),
                     f"AMBI encoded with {who}, profile {prof.name}"]
            canvas = compose(panels, title, bool(zs), zs, zpos, max(1, a.cols))
            name = cmp_dir / f"{stem}_lam{'-'.join(f'{l:g}' for l in a.lam)}_compare.png"
            canvas.save(name)
            if crop_at:
                Image.fromarray(rgb).save(conv / f"{stem}_original.png")
            print(f"  wrote {name}  ({canvas.size[0]}x{canvas.size[1]})  | converted files in {conv}", flush=True)

        except Exception as e:
            failed.append(f.name)
            print(f"  FAILED {f.name}: {type(e).__name__}: {e}", flush=True)
    if failed:
        print(f"\n{len(failed)} picture(s) failed: " + ", ".join(failed))
    print(f"\nside-by-side pictures: {cmp_dir}\nconverted files:      {conv}")

if __name__ == "__main__":
    main()
