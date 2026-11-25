import argparse, dataclasses, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from ambi_core import CodecConfig, rgb_to_ycc, fit_entropy_model
from ambi_core.config import INTRA_PRED_NAMES
from ambi_core.calibrate import calibrate_mse_scale
from ambi_core.profile import Profile
from gen_oracle_data import iter_dir_images, extract_tiles

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--id", type=int, required=True, help="profile id stored in .ambi headers (0..255; demo profile uses 1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-images", type=int, default=0, help="use a random subset of this many images (0 = all)")
    ap.add_argument("--tiles", type=int, default=48, help="tiles used for fitting (more = better tables, slower)")
    ap.add_argument("--per-image", type=int, default=2, help="max tiles per image, so many different photos are used")
    ap.add_argument("--flat-keep", type=float, default=1.0,
                    help="screenshots: how likely a flat (blank) tile is to be picked relative to a tile with content (1 = equally, 0.15 = 15%%)")
    ap.add_argument("--fit-lam", type=float, default=0.03)
    ap.add_argument("--fit-iters", type=int, default=3)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=1, help="parallel processes for the fit (e.g. your logical core count)")
    ap.add_argument("--intra", nargs="*", default=[], choices=INTRA_PRED_NAMES, metavar="PRED",
                    help="intra predictors to use (" + " ".join(INTRA_PRED_NAMES) + "); none = no intra prediction")
    ap.add_argument("--base-profile", help="copy the config of this profile (e.g. the same domain without intra) instead of the defaults")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg0 = CodecConfig()
    if a.base_profile:
        cfg0 = dataclasses.replace(Profile.load(a.base_profile).cfg, intra_preds=tuple(a.intra))
    else:
        cfg0 = dataclasses.replace(cfg0, intra_preds=tuple(a.intra))
    jobs = extract_tiles(iter_dir_images(a.image_dir, a.max_images, a.seed), a.per_image, cfg0.tile, a.seed, a.flat_keep)
    if not jobs:
        sys.exit(f"no usable images (or no image >= {cfg0.tile} px) in {a.image_dir}")
    rng = np.random.default_rng(a.seed)
    if len(jobs) > a.tiles:
        jobs = [jobs[i] for i in sorted(rng.choice(len(jobs), a.tiles, replace=False))]
    tiles = [np.ascontiguousarray(rgb_to_ycc(j[3])) for j in jobs]
    print(f"{len(tiles)} tiles from {len({j[0] for j in jobs})} images", flush=True)

    t0 = time.time()
    if a.base_profile:
        cfg = cfg0
        print(f"config copied from {a.base_profile} (mse_scale {cfg.mse_scale:.1f}), intra predictors: {list(cfg.intra_preds) or 'none'}", flush=True)
    else:
        c = calibrate_mse_scale(cfg0, tiles[:min(8, len(tiles))])
        cfg = dataclasses.replace(cfg0, mse_scale=c)
        print(f"calibrated mse_scale c = {c:.1f}, intra predictors: {list(cfg.intra_preds) or 'none'}", flush=True)
    model = fit_entropy_model(cfg, tiles, a.fit_lam, iters=a.fit_iters, mu_bits=a.mu, workers=a.workers)
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    prof = Profile(a.name, a.id, cfg, model)
    prof.save(out)
    print(f"wrote {out}  (fingerprint {prof.fingerprint:08x}, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
