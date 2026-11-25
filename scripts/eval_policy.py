import argparse, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from PIL import Image
from ambi_core import psnr_rgb
from ambi_core.bitstream import decode_image, encode_image
from ambi_core.metrics import ssim_y
from ambi_core.profile import Profile
from ambi_core.rects import RectSpace
from ambi_core.search import TileTables
from ambi_rl.baselines import best_uniform_gap
from ambi_rl.data import OracleData
from ambi_rl.partitioner import PolicyPartitioner
from ambi_rl.train import Trainer

ap = argparse.ArgumentParser()
ap.add_argument("policy"); ap.add_argument("--val-dir"); ap.add_argument("--images", nargs="*", default=[])
ap.add_argument("--profile", default="profiles/photo-demo.ambiprofile")
ap.add_argument("--lams", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
ap.add_argument("--device", default="cpu"); ap.add_argument("--json")
a = ap.parse_args()
prof = Profile.load(a.profile)
part = PolicyPartitioner.load(a.policy, a.device)
res = {}
if a.val_dir:
    space = RectSpace(prof.cfg)
    val = OracleData.load(a.val_dir, space, a.device, prof.fingerprint)
    tr = Trainer(part.model, space, val, a.device)
    ev, gaps = tr.evaluate(val)
    print(f"J gap to the optimum on {val.n} held-out tiles (mean / median / max % of per-tile gaps, fraction exactly optimal, AGGREGATE %)")
    for l, v in ev.items():
        print(f"  lambda {l:<6g} {v[0]:7.2f} {v[1]:6.2f} {v[2]:8.2f}   {v[3]:.2f}   {v[4]:6.2f}")
    res["j_gap"] = {str(l): v for l, v in ev.items()}
    ug = best_uniform_gap(space, val)
    opt = val.root_opt.cpu().numpy()
    ug_agg = (ug / 100 * opt).sum(0) / opt.sum(0) * 100
    print("reference: best fixed grid of equal blocks (8..64 px, chosen per tile with hindsight), aggregate gap " +
          "  ".join(f"{float(l):g}:{ug_agg[i]:.1f}%" for i, l in enumerate(ev)))
    res["uniform_grid_mean_gap"] = {str(l): float(ug[:, i].mean()) for i, l in enumerate(ev)}
    res["uniform_grid_aggregate_gap"] = {str(l): float(ug_agg[i]) for i, l in enumerate(ev)}
    tiles = val.ycc[:64].cpu().numpy()
    t0 = time.time(); part.trees(tiles, 0.05); dt = time.time() - t0
    print(f"policy partitioning: {dt / len(tiles) * 1000:.1f} ms per tile ({a.device}, batch {part.batch}); oracle ~4000 ms per tile")
    res["policy_ms_per_tile"] = dt / len(tiles) * 1000
for path in a.images:
    rgb = np.array(Image.open(path).convert("RGB")); H, W = rgb.shape[:2]
    print(f"\n{path} ({W}x{H}): real .ambi files, oracle vs policy (same lambda, deblock auto)")
    tabs = TileTables(rgb, prof)
    print(f"  {'lambda':>7} | {'oracle bytes':>12} {'PSNR':>7} {'SSIM':>7} | {'policy bytes':>12} {'PSNR':>7} {'SSIM':>7} | {'size':>7} {'dPSNR':>7}")
    rows = []
    for lam in a.lams:
        o = encode_image(rgb, prof, lam, tables=tabs.tables); od = decode_image(o.data, prof)
        t0 = time.time(); p = encode_image(rgb, prof, lam, partitioner=part); pdt = time.time() - t0
        pd = decode_image(p.data, prof)
        r = dict(lam=lam, oracle_bytes=o.nbytes, oracle_psnr=psnr_rgb(rgb, od), oracle_ssim=ssim_y(rgb, od),
                 policy_bytes=p.nbytes, policy_psnr=psnr_rgb(rgb, pd), policy_ssim=ssim_y(rgb, pd), policy_encode_s=pdt)
        rows.append(r)
        print(f"  {lam:>7g} | {r['oracle_bytes']:>12} {r['oracle_psnr']:>7.2f} {r['oracle_ssim']:>7.4f} | "
              f"{r['policy_bytes']:>12} {r['policy_psnr']:>7.2f} {r['policy_ssim']:>7.4f} | "
              f"{100 * (r['policy_bytes'] / r['oracle_bytes'] - 1):>+6.1f}% {r['policy_psnr'] - r['oracle_psnr']:>+7.2f}")
    res[path] = rows
if a.json:
    pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
