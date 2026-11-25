import argparse, json, pathlib, queue, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from ambi_core import psnr_rgb
from ambi_core.bitstream import decode_image, encode_image
from ambi_core.metrics import ssim_y
from ambi_core.bitstream import tile_grid
from ambi_core.profile import Profile
from ambi_core.parallel import run_tasks
from ambi_core.progress import LiveLine, fmt_time
from ambi_core.search import MAX_TABLE_TILES, SAMPLE_TILES, TileTables
from side_by_side import load_rgb


ENC_WEIGHT = 3.0


def work_units(rgb, prof, oracle, n_lams):
    if not oracle:
        return 0, 0.0, 1.0
    ty, tx = tile_grid(*rgb.shape[:2], prof.cfg.tile)
    n = ty * tx
    full = n <= max(16, MAX_TABLE_TILES * 9 // prof.cfg.n_modes)
    if full:
        return n, float(prof.cfg.n_modes), ENC_WEIGHT
    return 0, 0.0, float(n * prof.cfg.n_modes)


class Bar:
    def __init__(self, live, head, total, width=24):
        self.live, self.head, self.total, self.width = live, head, max(float(total), 1e-9), width
        self.done, self.t0 = 0.0, time.time()

    def fraction(self, within=0.0):
        return min(1.0, (self.done + within) / self.total)

    def render(self, stage, within=0.0):
        f = self.fraction(within)
        n = int(f * self.width)
        el = time.time() - self.t0
        eta = f" ~{fmt_time(el * (1 - f) / f)} left" if f > 0.03 else ""
        self.live.update(f"{self.head} [{'#' * n}{'-' * (self.width - n)}] {100 * f:3.0f}%  {stage}  {fmt_time(el)}{eta}")


def curve(rgb, prof, part, oracle, lams, bar=None, tag=""):
    ntab, w_tile, w_enc = work_units(rgb, prof, oracle, len(lams))
    tabs = None
    if oracle and ntab:
        tabs = TileTables(rgb, prof, progress=(lambda n, tot: bar.render(f"{tag} tables {n}/{tot}", n * w_tile)) if bar else None)
        if bar:
            bar.done += ntab * w_tile
    out = []
    for k, lam in enumerate(lams):
        if bar:
            bar.render(f"{tag} encode {k + 1}/{len(lams)}")
        r = (encode_image(rgb, prof, lam, tables=tabs.tables if tabs is not None else None) if oracle
             else encode_image(rgb, prof, lam, partitioner=part))
        dec = decode_image(r.data, prof)
        out.append((lam, r.nbytes, psnr_rgb(rgb, dec), ssim_y(rgb, dec)))
        if bar:
            bar.done += w_enc
    return out


_PROFILES = {}


def load_profile_cached(path):
    if path not in _PROFILES:
        _PROFILES[path] = Profile.load(path)
    return _PROFILES[path]


_Q = None


def init_worker(q):
    global _Q
    _Q = q


class QueueBar:
    def __init__(self, key):
        self.key, self.done, self._t = key, 0.0, 0.0

    def render(self, stage, within=0.0):
        now = time.time()
        if _Q is not None and now - self._t > 0.5:
            self._t = now
            _Q.put((self.key, self.done + within))


def drain(q, prog):
    while True:
        try:
            key, v = q.get_nowait()
        except queue.Empty:
            return
        prog[key] = max(prog.get(key, 0.0), v)


def curve_task(args):
    pi, side, rgb, path, lams = args
    t0 = time.time()
    return pi, side, curve(rgb, load_profile_cached(path), None, True, lams, QueueBar((pi, side)), ""), time.time() - t0


def bytes_at(points, key, value):
    x = np.array([p[key] for p in points]); lb = np.log([p[1] for p in points])
    o = np.argsort(x, kind="stable"); x, lb = x[o], lb[o]
    if value < x[0] or value > x[-1]:
        return None
    return float(np.exp(np.interp(value, x, lb)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--a", required=True, help="profile A (the reference)")
    ap.add_argument("--b", required=True, help="profile B (compared against A)")
    ap.add_argument("--policy-a"); ap.add_argument("--policy-b")
    ap.add_argument("--oracle", action="store_true", help="exact partitioner for both (no policy files needed)")
    ap.add_argument("--prune-k", type=int, default=0, metavar="K",
                    help="exact search over each policy's K best actions per rectangle instead of the policy alone (3 is a good start)")
    ap.add_argument("--crop", type=int, default=0, help="crop side in px (multiple of 64); 0 = whole pictures")
    ap.add_argument("--crops-per-image", type=int, default=1)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.01, 0.02, 0.04, 0.08, 0.16, 0.32])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes (needs --oracle): one picture-and-profile job per process, e.g. 16; default 1")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.workers > 1 and not a.oracle:
        sys.exit("--workers needs --oracle (a policy run uses one process and the GPU)")
    if not a.oracle and not (a.policy_a and a.policy_b):
        sys.exit("give --policy-a and --policy-b, or --oracle")
    pa, pb = Profile.load(a.a), Profile.load(a.b)
    parts = [None, None]
    if not a.oracle:
        from ambi_rl.pruned import make_partitioner
        parts = [make_partitioner(a.policy_a, pa, a.device, a.prune_k), make_partitioner(a.policy_b, pb, a.device, a.prune_k)]
    files = [f for f in sorted(pathlib.Path(a.image_dir).rglob("*"))
             if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")]
    if a.max_images:
        files = files[:a.max_images]
    rng = np.random.default_rng(a.seed)
    jobs = []                                                          # (file, y, x, rgb): the pictures to compare, in order
    for f in files:
        full, _ = load_rgb(f)
        H, W = full.shape[:2]
        if a.crop and (H < a.crop or W < a.crop):
            continue
        if not a.crop:
            jobs.append((f, 0, 0, full))
        for _ in range(a.crops_per_image if a.crop else 0):
            y = int(rng.integers(0, (H - a.crop) // 64 + 1)) * 64
            x = int(rng.integers(0, (W - a.crop) // 64 + 1)) * 64
            jobs.append((f, y, x, np.ascontiguousarray(full[y:y + a.crop, x:x + a.crop])))
    n_total, t0, live = len(jobs), time.time(), LiveLine()
    per_pic = [None] * n_total

    def weight(rgb, pr):
        nt, wt, we = work_units(rgb, pr, a.oracle, len(a.lams))
        return nt * wt + len(a.lams) * we

    if a.workers > 1 and n_total:
        tasks = []                                                     # (weight, args): slowest first, so no big job starts last
        for pi, (f, y, x, rgb) in enumerate(jobs):
            for side, pr, path in ((0, pa, a.a), (1, pb, a.b)):
                tasks.append((weight(rgb, pr), (pi, side, rgb, path, a.lams)))
        tasks.sort(key=lambda t: -t[0])
        running = min(a.workers, len(tasks))
        print(f"{n_total} pictures x 2 profiles = {len(tasks)} jobs on {running} worker processes", flush=True)
        import multiprocessing as mp
        q = mp.get_context("spawn").Queue()
        wpic = {pi: weight(jobs[pi][3], pa) + weight(jobs[pi][3], pb) for pi in range(n_total)}
        prog, bar = {}, Bar(live, "all pictures", sum(wpic.values()))
        got, n_done, n_pics = {}, 0, 0
        for i, res in run_tasks(curve_task, [t for _, t in tasks], a.workers, tick=0.5, initializer=init_worker, initargs=(q,)):
            if res is not None:
                pi, side, cv, secs = res
                got.setdefault(pi, {})[side] = (cv, secs)
                prog[(pi, side)] = tasks[i][0]
                n_done += 1
                if len(got[pi]) == 2:
                    f, y, x, _ = jobs[pi]
                    per_pic[pi] = dict(image=f.name, y=y, x=x, a=got[pi][0][0], b=got[pi][1][0])
                    n_pics += 1
                    live.clear()
                    print(f"  {n_pics}/{n_total} {f.name}  (A {fmt_time(got[pi][0][1])}, B {fmt_time(got[pi][1][1])}, "
                          f"total {fmt_time(time.time() - t0)})", flush=True)
            drain(q, prog)
            bar.done = sum(prog.values())
            per = " ".join(f"{pi + 1}:{min(100, 100 * (prog.get((pi, 0), 0) + prog.get((pi, 1), 0)) / wpic[pi]):3.0f}%" for pi in range(n_total))
            bar.render(f"{n_done}/{len(tasks)} jobs | pictures {per}")
        live.clear()
    else:
        pic_secs = []
        for pi, (f, y, x, rgb) in enumerate(jobs):
            t_pic = time.time()
            nm = f.name if len(f.name) <= 16 else f.name[:8] + "..." + f.suffix
            bar = Bar(live, f"pic {pi + 1}/{n_total} {nm}", weight(rgb, pa) + weight(rgb, pb))
            ca = curve(rgb, pa, parts[0], a.oracle, a.lams, bar, "A")
            cb = curve(rgb, pb, parts[1], a.oracle, a.lams, bar, "B")
            per_pic[pi] = dict(image=f.name, y=y, x=x, a=ca, b=cb)
            pic_secs.append(time.time() - t_pic)
            live.clear()
            left = (n_total - pi - 1) * float(np.mean(pic_secs))
            print(f"  {pi + 1}/{n_total} {f.name}  (picture {fmt_time(pic_secs[-1])}, total {fmt_time(time.time() - t0)}"
                  f"{f', about {fmt_time(left)} left' if left > 0 else ''})", flush=True)
    per_pic = [p for p in per_pic if p is not None]

    print(f"\nB = {pathlib.Path(a.b).name}  against  A = {pathlib.Path(a.a).name}   ({len(per_pic)} pictures; positive = B file bigger, negative = B smaller)")
    for key, name in ((2, "PSNR"), (3, "SSIM(Y)")):
        print(f"\nat equal {name}, one row per B lambda (pictures whose quality falls inside A's range only)")
        print(f"  {'lambda':>7} {'pics':>5} {'B ' + name:>11} {'B bpp':>7} {'pooled':>8} {'median':>8} {'B smaller in':>13}")
        for li, lam in enumerate(a.lams):
            ratios, bb, aa, q, px = [], 0.0, 0.0, [], 0
            for p in per_pic:
                pt = p["b"][li]
                ab = bytes_at(p["a"], key, pt[key])
                if ab is None:
                    continue
                ratios.append(pt[1] / ab); bb += pt[1]; aa += ab; q.append(pt[key])
                px += a.crop * a.crop if a.crop else 0
            if not ratios:
                print(f"  {lam:>7g} {0:>5}   (B's quality is outside A's range)")
                continue
            r = np.array(ratios)
            print(f"  {lam:>7g} {len(r):>5} {np.mean(q):>11.3f} {'':>7} {100 * (bb / aa - 1):>+7.1f}% {100 * (np.median(r) - 1):>+7.1f}% {100 * np.mean(r < 1):>12.0f}%")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(per_pic, indent=1))


if __name__ == "__main__":
    main()
