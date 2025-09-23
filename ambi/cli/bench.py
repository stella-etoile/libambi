# ambi/cli/bench.py
from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm.auto import tqdm

from ambi.codec import encode_image, decode_image
from ambi.io import load_image_rgb
from ambi.distortion import psnr, ssim, msssim

def _is_image(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _gather_inputs(root: Path) -> List[Tuple[Path, Path, str]]:
    """
    Returns list of (src_img, out_ambi, out_rec_png)
    """
    out = []
    if root.is_file() and _is_image(root):
        img = root
        out_dir = root.parent
        out_ambi = out_dir / (root.stem + ".ambi")
        out_png  = out_dir / (root.stem + ".rec.png")
        out.append((img, out_ambi, out_png))
    else:
        imgs = sorted([p for p in root.rglob("*") if _is_image(p)])
        for img in imgs:
            out_dir = img.parent
            out_ambi = out_dir / (img.stem + ".ambi")
            out_png  = out_dir / (img.stem + ".rec.png")
            out.append((img, out_ambi, out_png))
    return out

def _load_cfg(cfg_path: Path | None) -> dict:
    if cfg_path is None:
        return {}
    import yaml
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}

def main():
    ap = argparse.ArgumentParser(description="AMBI bench: encode+decode+metrics")
    ap.add_argument("input", type=str, help="image file or directory")
    ap.add_argument("--config", type=str, default=None, help="path to config.yaml")
    ap.add_argument("--workers", type=int, default=None, help="override decode workers")
    ap.add_argument("--chunk", type=int, default=None, help="override decode chunk")
    ap.add_argument("--metric-on", type=str, default="y", choices=["y", "rgb"], help="run SSIM/MS-SSIM on: y or rgb")
    ap.add_argument("--keep", action="store_true", help="keep .ambi files (otherwise removed after bench)")
    args = ap.parse_args()

    root = Path(args.input)
    cfg = _load_cfg(Path(args.config)) if args.config else {}

    tasks = _gather_inputs(root)
    if not tasks:
        print("No images found.")
        return 0

    rows = []
    for src, ambi, rec in tqdm(tasks, desc="AMBI bench", unit="img"):
        # encode
        t0 = time.perf_counter()
        encode_image(src, ambi, cfg)
        t1 = time.perf_counter()

        # decode
        t2 = time.perf_counter()
        decode_image(ambi, rec, num_workers=args.workers, chunk_size=args.chunk)
        t3 = time.perf_counter()

        # metrics
        ref = load_image_rgb(src)
        out = load_image_rgb(rec)
        h, w = ref.shape[:2]
        bits = ambi.stat().st_size * 8
        bpp = bits / (w * h)

        row = {
            "name": src.name,
            "WxH": f"{w}x{h}",
            "bpp": round(bpp, 4),
            "PSNR(dB)": round(psnr(ref, out), 3),
            "SSIM": round(ssim(ref, out, on=args.metric_on), 5),
            "MS-SSIM": round(msssim(ref, out, on=args.metric_on), 5),
            "t_enc(s)": round(t1 - t0, 4),
            "t_dec(s)": round(t3 - t2, 4),
            "size(bytes)": ambi.stat().st_size,
            "ambi": str(ambi),
            "rec": str(rec),
        }
        rows.append(row)

        if not args.keep:
            try: ambi.unlink()
            except Exception: pass

    # pretty print table
    # narrow summary first
    print("\n=== AMBI bench summary ===")
    hdr = ["name", "WxH", "bpp", "PSNR(dB)", "SSIM", "MS-SSIM", "t_enc(s)", "t_dec(s)", "size(bytes)"]
    colw = {h: max(len(h), max((len(str(r[h])) for r in rows), default=0)) for h in hdr}
    line = " | ".join(h.ljust(colw[h]) for h in hdr)
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(r[h]).ljust(colw[h]) for h in hdr))

    # return non-zero if any decode visibly failed (optional—here always 0)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())