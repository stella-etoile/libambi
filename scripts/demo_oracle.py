from __future__ import annotations

import argparse, io, json, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import Image
from skimage import data
from skimage.metrics import structural_similarity

from ambi_core import (CodecConfig, EntropyModel, LeafCoder, rgb_to_ycc, ycc_to_rgb, psnr_rgb,
                       solve_tile, reconstruct_tile, fit_entropy_model)
from ambi_core.distortion import SsimContext
from ambi_core.oracle import Leaf, pad_luma
from ambi_core.calibrate import calibrate_mse_scale
from ambi_core.viz import draw_partition


def crop_tiles(ycc, rgb_shape, tile, y0, x0, ny, nx):
    return [[ycc[:, y0 + j*tile: y0 + (j+1)*tile, x0 + i*tile: x0 + (i+1)*tile].copy()
             for i in range(nx)] for j in range(ny)]


def encode_crop(cfg, model, ycc_crop, lam, mu):
    n_y, n_x = ycc_crop.shape[1] // cfg.tile, ycc_crop.shape[2] // cfg.tile
    recon = np.zeros_like(ycc_crop); leaves = []; bits = 0.0
    for j in range(n_y):
        for i in range(n_x):
            t = ycc_crop[:, j*cfg.tile:(j+1)*cfg.tile, i*cfg.tile:(i+1)*cfg.tile]
            res = solve_tile(cfg, model, t, lam, mu)
            rc, _ = reconstruct_tile(cfg, model, t, res.leaves)
            recon[:, j*cfg.tile:(j+1)*cfg.tile, i*cfg.tile:(i+1)*cfg.tile] = rc
            bits += res.total_bits
            leaves += [Leaf(l.x + i*cfg.tile, l.y + j*cfg.tile, l.w, l.h, l.mode, l.bits, l.dist) for l in res.leaves]
    return recon, leaves, bits


def jpeg_points(rgb, qualities=(20, 30, 40, 50, 60, 70, 80, 90)):
    out = []
    h, w = rgb.shape[:2]
    for q in qualities:
        buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "JPEG", quality=q, optimize=True)
        base = io.BytesIO(); Image.fromarray(np.full((8, 8, 3), 128, np.uint8)).save(base, "JPEG", quality=q, optimize=True)
        dec = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
        bpp = (len(buf.getvalue()) - len(base.getvalue())) * 8 / (h * w)
        out.append({"quality": q, "bpp": bpp, "psnr": psnr_rgb(rgb, dec), "ssim": ssim_rgb(rgb, dec)})
    return out


def ssim_rgb(a, b):
    ya = np.array(Image.fromarray(a).convert("L")); yb = np.array(Image.fromarray(b).convert("L"))
    return float(structural_similarity(ya, yb, data_range=255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out"); ap.add_argument("--train-tiles", type=int, default=24)
    ap.add_argument("--fit-iters", type=int, default=2); ap.add_argument("--fit-lam", type=float, default=0.03)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    ap.add_argument("--mu", type=float, default=0.0)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(exist_ok=True)
    cfg = CodecConfig(); T = cfg.tile
    rng = np.random.default_rng(0)

    train_imgs = [data.coffee(), data.rocket(), data.hubble_deep_field()]
    tiles = []
    for k in range(a.train_tiles):
        im = train_imgs[k % len(train_imgs)]
        y = int(rng.integers(0, im.shape[0] - T)); x = int(rng.integers(0, im.shape[1] - T))
        tiles.append(rgb_to_ycc(im[y:y+T, x:x+T]))
    t0 = time.time()
    c = calibrate_mse_scale(cfg, tiles[:6])
    cfg = CodecConfig(mse_scale=c)
    print(f"calibrated mse_scale c = {c:.1f}")
    print(f"fitting entropy model on {len(tiles)} tiles ...")
    model = fit_entropy_model(cfg, tiles, a.fit_lam, iters=a.fit_iters, mu_bits=a.mu)
    model.save(out / "model_demo.npz")
    print(f"fit time {time.time()-t0:.0f}s")

    tests = {"astronaut": data.astronaut()[40:232, 160:352], "cat": data.chelsea()[20:212, 100:292]}
    report = {}
    for name, rgb in tests.items():
        ycc = rgb_to_ycc(rgb)
        rows = []
        for lam in a.lams:
            t0 = time.time()
            recon, leaves, bits = encode_crop(cfg, model, ycc, lam, a.mu)
            dec = ycc_to_rgb(recon)
            bpp = bits / (rgb.shape[0] * rgb.shape[1])
            modes = {}
            for l in leaves:
                modes[cfg.mode_family(l.mode)] = modes.get(cfg.mode_family(l.mode), 0) + l.w * l.h
            row = {"lam": lam, "bpp": bpp, "psnr": psnr_rgb(rgb, dec), "ssim": ssim_rgb(rgb, dec),
                   "n_leaves": len(leaves), "mean_leaf_area": float(np.mean([l.w * l.h for l in leaves])),
                   "pixel_share_by_family": {k: v / (rgb.shape[0] * rgb.shape[1]) for k, v in modes.items()},
                   "sec": time.time() - t0}
            rows.append(row)
            print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
            draw_partition(rgb, leaves, cfg).save(out / f"{name}_partition_lam{lam}.png")
            Image.fromarray(dec).save(out / f"{name}_recon_lam{lam}.png")
        Image.fromarray(rgb).save(out / f"{name}_orig.png")
        report[name] = {"ambi": rows, "jpeg": jpeg_points(rgb)}
        print(name, "JPEG:", [(p["quality"], round(p["bpp"], 3), round(p["psnr"], 2)) for p in report[name]["jpeg"]])
    (out / "report.json").write_text(json.dumps({"mse_scale": c, "report": report}, indent=1))


if __name__ == "__main__":
    main()
