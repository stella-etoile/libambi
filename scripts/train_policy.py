import argparse, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from ambi_core.profile import Profile
from ambi_core.rects import RectSpace
from ambi_rl.data import OracleData, TileStore
from ambi_rl.model import PolicyNet
from ambi_rl.partitioner import save_policy
from ambi_rl.train import Trainer, fit

ap = argparse.ArgumentParser()
ap.add_argument("--train-dir", required=True); ap.add_argument("--val-dir")
ap.add_argument("--profile", default="profiles/photo-demo.ambiprofile")
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="auto")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--width", type=int, default=64); ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--tiles-per-step", type=int, default=8); ap.add_argument("--lams-per-tile", type=int, default=2)
ap.add_argument("--uniform-per", type=int, default=48)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--temp-bits", type=float, default=4.0)
ap.add_argument("--regret-weight", type=float, default=1.0, help="weight of the expected-regret term (0 = plain imitation)")
ap.add_argument("--norm-floor", type=float, default=0.25, help="floor on the tile J used to normalise the regret term, as a fraction of the median tile J")
ap.add_argument("--warmup-steps", type=int, default=200, help="steps before the policy's own rollouts join the batch")
ap.add_argument("--eval-every", type=int, default=250)
ap.add_argument("--limit-tiles", type=int); ap.add_argument("--threads", type=int)
ap.add_argument("--store", choices=["auto", "gpu", "disk"], default="auto",
                help="where the cost arrays live: gpu = all on the device (about 1 MB per tile of device memory), disk = read from a cache file next to "
                     "the tiles, only the tiles a step needs go to the device (any data size), auto = disk when a set needs more than --gpu-budget-gb")
ap.add_argument("--gpu-budget-gb", type=float, default=6.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device)
if a.threads:
    torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
prof = Profile.load(a.profile)
space = RectSpace(prof.cfg)
t0 = time.time()
train = OracleData.load(a.train_dir, space, dev, prof.fingerprint, a.limit_tiles, a.store, a.gpu_budget_gb)
val = OracleData.load(a.val_dir, space, dev, prof.fingerprint, None, a.store, a.gpu_budget_gb) if a.val_dir else None
print(f"device {dev}; {train.n} train tiles x {train.n_lams} lambdas" + (" (on disk)" if isinstance(train.best, TileStore) else "") + (f"; {val.n} val tiles x {val.n_lams} lambdas" if val else "")
      + f"  (loaded in {time.time() - t0:.0f}s)", flush=True)
model = PolicyNet(prof.cfg, a.width, a.hidden)
print("parameters:", sum(p.numel() for p in model.parameters()), flush=True)
tr = Trainer(model, space, train, dev, a.lr, a.temp_bits, a.tiles_per_step, a.lams_per_tile, a.uniform_per, a.seed,
             a.regret_weight, norm_floor=a.norm_floor)
out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
best = {"gap": float("inf")}
best_path = out.with_name(out.stem + "_best" + out.suffix)


def on_eval(it, ev):
    save_policy(model, out, {"steps_so_far": it})
    gap = sum(v[4] for v in ev.values()) / len(ev)
    if gap < best["gap"]:
        best["gap"] = gap
        save_policy(model, best_path, {"steps_so_far": it, "val_aggregate_gap_pct": gap})
        print(f"  new best held-out gap {gap:.2f}% -> {best_path}", flush=True)


hist = fit(tr, a.steps, log_every=25, eval_every=a.eval_every, val=val, warmup_steps=a.warmup_steps, on_eval=on_eval)
final = {}
for name, ds in (("train", train), ("val", val)):
    if ds is None:
        continue
    print(f"evaluating {name} tiles (a minute or two) ...", flush=True)
    ev, _ = tr.evaluate(ds)
    final[name] = {str(l): {"mean_gap_pct": v[0], "median_gap_pct": v[1], "max_gap_pct": v[2], "frac_optimal": v[3], "aggregate_gap_pct": v[4]} for l, v in ev.items()}
    print(f"{name}: J gap to the optimum, per lambda (mean / median / max % of per-tile gaps, fraction exactly optimal, AGGREGATE %)")
    for l, v in ev.items():
        print(f"  lambda {l:<6g} {v[0]:7.2f} {v[1]:6.2f} {v[2]:8.2f}   {v[3]:.2f}   {v[4]:6.2f}")
save_policy(model, out, {"steps": a.steps, "train_tiles": train.n, "final": final, "profile_fingerprint": prof.fingerprint})
(out.with_suffix(".json")).write_text(json.dumps({"args": vars(a), "final": final, "history": hist}, indent=1))
print("saved", out, f"({time.time() - t0:.0f}s total)")
