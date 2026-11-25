import argparse, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from ambi_core.profile import Profile
from ambi_core.rects import RectSpace
from ambi_rl.data import OracleData
from ambi_rl.partitioner import load_policy
from ambi_rl.topk import diagnose

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data-dir", required=True, help="oracle data (scripts/gen_oracle_data.py output), normally the validation set")
ap.add_argument("--profile", required=True); ap.add_argument("--policy", required=True)
ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12])
ap.add_argument("--limit-tiles", type=int, default=150, help="first N tiles of the data set (default 150)")
ap.add_argument("--batch", type=int, default=8, help="tiles per network pass (lower it if the GPU runs out of memory)")
ap.add_argument("--chunk", type=int, default=2048, help="rectangles per head call (lower it if the GPU runs out of memory)")
ap.add_argument("--device", default="auto"); ap.add_argument("--json")
a = ap.parse_args()

dev = "cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device)
prof = Profile.load(a.profile)
space = RectSpace(prof.cfg)
data = OracleData.load(a.data_dir, space, dev, prof.fingerprint, a.limit_tiles, "gpu")
model = load_policy(a.policy, dev)
print(f"{data.n} tiles x {data.n_lams} lambdas, policy {pathlib.Path(a.policy).name}, device {dev}", flush=True)
t0 = time.time()
res = diagnose(model, space, data, a.ks, dev, a.batch, a.chunk)
ks = sorted(next(iter(res.values())))
lams = list(res)
print(f"\nJ gap to the exact oracle when the DP only considers the policy's top k actions per rectangle (aggregate %, {data.n} tiles)")
print("      k  " + "".join(f"{l:>8g}" for l in lams) + "     mean")
for k in ks:
    g = [res[l][k]["gap_pct"] for l in lams]
    print(f"  {k:>5}  " + "".join(f"{x:8.2f}" for x in g) + f"  {sum(g) / len(g):7.2f}")
print("\nwork and coverage, mean over lambdas")
print("      k   rectangles visited   oracle choice in top k   tiles exactly optimal")
for k in ks:
    v = sum(res[l][k]["visited_frac"] for l in lams) / len(lams)
    h = sum(res[l][k]["path_recall"] for l in lams) / len(lams)
    o = sum(res[l][k]["frac_optimal"] for l in lams) / len(lams)
    print(f"  {k:>5}   {v * 100:17.2f}%   {h * 100:21.1f}%   {o * 100:20.1f}%")
print("\nk = 1 is the policy's own greedy tree; the oracle visits 100% of the rectangles.")
print("'oracle choice in top k' is measured on the rectangles of the oracle's optimal tree.")
print(f"({time.time() - t0:.0f}s)")
if a.json:
    pathlib.Path(a.json).write_text(json.dumps({str(l): {str(k): v for k, v in res[l].items()} for l in lams}, indent=1))
