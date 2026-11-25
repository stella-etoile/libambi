import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one process per core: BLAS threads on top would only oversubscribe the CPU
import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from multiprocessing import Pool
from PIL import Image

from ambi_core import rgb_to_ycc
from ambi_core.oracle import build_tile_table, solve_table_values
from ambi_core.profile import Profile, default_profile
from ambi_core.progress import LiveLine, fmt_time
from ambi_core.rects import RectSpace

DEFAULT_LAMS = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
_W = {}


def stock_images(which):
    from skimage import data as d
    g3 = lambda a: np.stack([a] * 3, -1)
    if which == "train":
        return {"coffee": d.coffee(), "rocket": d.rocket(), "hubble": d.hubble_deep_field()[100:800, 100:900],
                "retina": d.retina()[300:1100, 300:1100], "brick": g3(d.brick()), "grass": g3(d.grass()),
                "gravel": g3(d.gravel()), "moon": g3(d.moon()), "camera": g3(d.camera()),
                "immuno": d.immunohistochemistry()}
    return {"astronaut": d.astronaut(), "chelsea": d.chelsea()}


def list_images(path, max_images=0, seed=0):
    """Image files under `path`; with max_images, a random subset (chosen before any image is decoded)."""
    files = [f for f in sorted(pathlib.Path(path).rglob("*"))
             if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")]
    if max_images and len(files) > max_images:
        rng = np.random.default_rng(seed)
        files = [files[i] for i in sorted(rng.choice(len(files), max_images, replace=False))]
    return files


def iter_dir_images(path, max_images=0, seed=0):
    """Yield (name, rgb) one image at a time (a big photo folder never has to fit in RAM)."""
    root = pathlib.Path(path)
    for f in list_images(path, max_images, seed):
        try:
            rgb = np.array(Image.open(f).convert("RGB"))
        except Exception as e:                                       # truncated or unreadable file: skip it
            print(f"  skipping {f}: {e}", flush=True)
            continue
        yield (f.stem if f.parent == root else f"{f.parent.name}/{f.stem}"), rgb


def extract_tiles(images, per_image, tile, seed, flat_keep=1.0, flat_std=2.0):
    """images: dict or iterable of (name, rgb). Returns [(name, y, x, rgb_tile)].
    flat_keep < 1 (screenshots: most tiles are blank background) makes a flat tile (pixel std < flat_std) flat_keep times as
    likely to be picked as any other tile when an image has more than per_image tiles. 1.0 = uniform, as before."""
    rng = np.random.default_rng(seed)
    jobs = []
    for name, rgb in (images.items() if hasattr(images, "items") else images):
        H, W = rgb.shape[:2]
        pos = [(y, x) for y in range(0, H - tile + 1, tile) for x in range(0, W - tile + 1, tile)]
        if per_image and len(pos) > per_image:
            if flat_keep < 1.0:
                w = np.array([1.0 if rgb[y:y + tile, x:x + tile].std() >= flat_std else flat_keep for (y, x) in pos])
                k = min(per_image, int((w > 0).sum()))
                idx = rng.choice(len(pos), k, replace=False, p=w / w.sum()) if k > 0 else []
            else:
                idx = rng.choice(len(pos), per_image, replace=False)
            pos = [pos[i] for i in sorted(idx)]
        for (y, x) in pos:
            jobs.append((name, y, x, np.ascontiguousarray(rgb[y:y + tile, x:x + tile])))
    return jobs


def _init(profile_path, lams):
    prof = Profile.load(profile_path) if profile_path else default_profile()
    _W.update(prof=prof, space=RectSpace(prof.cfg), lams=lams)


def tile_file_ok(path, src, fingerprint, lams):
    """True if `path` is a complete tile file for this very tile, profile and lambda list. A Ctrl+C can leave a truncated file
    (its zip index is missing, so opening it fails), and a rerun with other settings must not silently reuse old tiles."""
    try:
        with np.load(path) as z:
            return ("best" in z.files and str(z["src"]) == src and int(z["fingerprint"]) == int(fingerprint)
                    and np.array_equal(z["lams"], np.array(lams, np.float64)))
    except Exception:
        return False


def resume_stats(total, done, new, secs_since_mark):
    """(seconds per new tile, seconds left) for a run that may have resumed. Tiles that already existed finish instantly, so
    only the tiles computed in THIS run count towards the rate. (None, None) until a tile has been computed."""
    if new <= 0:
        return None, None
    rate = secs_since_mark / new
    return rate, rate * (total - done)


def _work(job):
    """Returns (index, seconds, skipped)."""
    i, name, y, x, rgb, path = job
    prof, space, lams = _W["prof"], _W["space"], _W["lams"]
    src = f"{name}:{y}:{x}"
    if path.exists() and tile_file_ok(path, src, prof.fingerprint, lams):
        return i, 0.0, True
    t0 = time.time()
    tile = np.ascontiguousarray(rgb_to_ycc(rgb))
    table = build_tile_table(prof.cfg, prof.model, tile)
    best = np.zeros((len(lams), space.n_rects), np.float32)
    leafc = np.zeros_like(best)
    for j, lam in enumerate(lams):
        b, l = solve_table_values(prof.cfg, table, lam)
        best[j], leafc[j] = space.flatten(b), space.flatten(l)
    tmp = path.with_name(path.stem + ".tmp.npz")                      # write beside it, then rename: never a half-written tile
    np.savez_compressed(tmp, rgb=rgb, best=best, leafc=leafc, lams=np.array(lams, np.float64),
                        src=np.array(src), fingerprint=np.array(prof.fingerprint, np.int64))
    os.replace(tmp, path)
    return i, time.time() - t0, False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stock", choices=["train", "val"], help="built-in skimage images (sandbox/demo)")
    g.add_argument("--image-dir", help="folder of images; tiles are cut on a 64 px grid")
    ap.add_argument("--profile", default="profiles/photo-demo.ambiprofile")
    ap.add_argument("--per-image", type=int, default=48, help="max tiles per image (random subset)")
    ap.add_argument("--max-images", type=int, default=0, help="use a random subset of this many images (0 = all)")
    ap.add_argument("--flat-keep", type=float, default=1.0,
                    help="screenshots: how likely a flat (blank) tile is to be picked relative to a tile with content (1 = equally, 0.15 = 15%%)")
    ap.add_argument("--lams", type=float, nargs="+", default=DEFAULT_LAMS)
    ap.add_argument("--extra-lams", type=float, nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2, help="worker processes (default: all logical cores)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    lams = sorted(set(a.lams) | set(a.extra_lams))
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prof = Profile.load(a.profile)
    images = stock_images(a.stock) if a.stock else iter_dir_images(a.image_dir, a.max_images, a.seed)
    tiles = extract_tiles(images, a.per_image, prof.cfg.tile, a.seed, a.flat_keep)
    jobs = [(i, n, y, x, rgb, out / f"tile_{i:05d}.npz") for i, (n, y, x, rgb) in enumerate(tiles)]
    print(f"{len(jobs)} tiles from {len({j[1] for j in jobs})} images, {len(lams)} lambdas {lams}, profile {prof.name}", flush=True)
    for stale in out.glob("*.tmp.npz"):                                  # left by an interrupted run
        stale.unlink()
    extra = [f for f in out.glob("tile_*.npz") if f.stem[5:].isdigit() and int(f.stem[5:]) >= len(jobs)]
    if extra:
        print(f"WARNING: {len(extra)} tile files in {out} have an index beyond this run's {len(jobs)} tiles (an earlier run with "
              f"other settings). Training would load them too: delete them or use a new --out.", flush=True)
    t0 = time.time(); done = new = skipped = 0
    t_mark = t0                                                          # last moment an already-existing tile was skipped
    live = LiveLine()
    with Pool(a.workers, initializer=_init, initargs=(a.profile, lams)) as pool:
        for i, dt, was_skipped in pool.imap_unordered(_work, jobs):
            done += 1
            now = time.time()
            if was_skipped:
                skipped += 1
                t_mark = now
            else:
                new += 1
            el = now - t0
            # the rate comes only from tiles computed in this run, and only once every worker has finished a tile or two
            ready = new >= min(2 * a.workers, len(jobs) - skipped)
            rate, left = resume_stats(len(jobs), done, new, now - t_mark) if ready else (None, None)
            tag = f"({skipped} already done) " if skipped else ""
            if live.enabled:
                live.update(f"{done}/{len(jobs)} tiles {100 * done // len(jobs)}%  {tag}"
                            + (f"{rate:.2f}s/tile  elapsed {fmt_time(el)}  ETA {fmt_time(left)}" if rate is not None
                               else f"elapsed {fmt_time(el)}  ETA ..."))
            elif done % 10 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} tiles {tag} {el:.0f}s elapsed" + (f"  ~{left:.0f}s left" if left is not None else ""), flush=True)
    live.done()
    print("done ->", out)
