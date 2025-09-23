import os, re, glob, argparse
from datetime import datetime

OUT_PATH = os.environ.get("OUT_PATH", "").strip()
EST_TOTAL_EPOCHS = int(os.environ.get("EST_TOTAL_EPOCHS", "200"))
NBATCH_DEFAULT = int(os.environ.get("NBATCH_DEFAULT", "32"))

RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"

PAT_BATCH_STATS = re.compile(
    r'^\[Epoch\s+(?P<epoch>\d+)\/(?P<epochs>\d+)\]\s+'
    r'(?P<batch>\d+)\/(?P<nbatch>~?\d+)\s+'
    r'(?P<kind>BATCH|TOTAL)\s+'
    r'bpp=\[(?P<bpp>[^\]]+)\]\s+'
    r'PSNR=\[(?P<psnr>[^\]]+)\]\s+'
    r'SSIM=(?P<ssim>[-+eE\d\.]+)\s+MS=(?P<msssim>[-+eE\d\.]+)\s+'
    r'R~?=(?P<R>[-+eE\d\.]+)\s*$'
)

PAT_QWEIGHTS = re.compile(
    r'^\[QWEIGHTS\]\s+'
    r'bpp_cuts=\(q1=(?P<bpp_q1>[-+eE\d\.]+),\s*med=(?P<bpp_med>[-+eE\d\.]+),\s*q3=(?P<bpp_q3>[-+eE\d\.]+)\)\s+'
    r'psnr_cuts=\(q1=(?P<psnr_q1>[-+eE\d\.]+),\s*med=(?P<psnr_med>[-+eE\d\.]+),\s*q3=(?P<psnr_q3>[-+eE\d\.]+)\)\s+'
    r'wb=\((?P<wb1>[-+eE\d\.]+),\s*(?P<wb2>[-+eE\d\.]+),\s*(?P<wb3>[-+eE\d\.]+)\)\s+'
    r'wp=\((?P<wp1>[-+eE\d\.]+),\s*(?P<wp2>[-+eE\d\.]+),\s*(?P<wp3>[-+eE\d\.]+)\)\s*$'
)

PAT_ADAPT = re.compile(
    r'^\[ADAPT\]\s+'
    r'bpp_tail_ema=(?P<bpp_tail_ema>[-+eE\d\.]+)\s+psnr_floor_ema=(?P<psnr_floor_ema>[-+eE\d\.]+)\s+'
    r'req_psnr=(?P<req_psnr>[-+eE\d\.]+)\s+req_bpp=(?P<req_bpp>[-+eE\d\.]+)\s+'
    r'd_psnr=(?P<d_psnr>[-+eE\d\.]+)\s+d_bpp=(?P<d_bpp>[-+eE\d\.]+)\s+'
    r'alpha_psnr:(?P<alpha_psnr_old>[-+eE\d\.]+)->(?P<alpha_psnr_new>[-+eE\d\.]+)\s+'
    r'alpha_bpp:(?P<alpha_bpp_old>[-+eE\d\.]+)->(?P<alpha_bpp_new>[-+eE\d\.]+)\s*$'
)

PAT_RL = re.compile(
    r'^\[RL\]\s+iter\s+(?P<iter>\d+)\/(?P<iters>\d+)\s+'
    r'bpp_med=(?P<bpp_med>[-+eE\d\.]+)\s+\(min=(?P<bpp_min>[-+eE\d\.]+),\s*q1=(?P<bpp_q1>[-+eE\d\.]+),\s*q3=(?P<bpp_q3>[-+eE\d\.]+),\s*max=(?P<bpp_max>[-+eE\d\.]+)\)\s+'
    r'PSNR_med=(?P<psnr_med>[-+eE\d\.]+)\s+dB\s+\(min=(?P<psnr_min>[-+eE\d\.]+),\s*q1=(?P<psnr_q1>[-+eE\d\.]+),\s*q3=(?P<psnr_q3>[-+eE\d\.]+),\s*max=(?P<psnr_max>[-+eE\d\.]+)\)\s+'
    r'SSIM=(?P<ssim>[-+eE\d\.]+)\s+MS-SSIM=(?P<msssim>[-+eE\d\.]+)\s+'
    r'Wbpp=(?P<Wbpp>[-+eE\d\.]+)\s+Wpsnr=(?P<Wpsnr>[-+eE\d\.]+)\s+mSSIM_med=(?P<msssim_med>[-+eE\d\.]+)\s+Wtotal=(?P<Wtotal>[-+eE\d\.]+)\s+'
    r'alpha_bpp=(?P<alpha_bpp>[-+eE\d\.]+)\s+alpha_psnr=(?P<alpha_psnr>[-+eE\d\.]+)\s*$'
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", str(s))

def newest_log():
    c = sorted(glob.glob("logs/*.out"), key=os.path.getmtime, reverse=True)
    return c[0] if c else None

def pct(a, b):
    if not b or b <= 0: return 0.0
    return 100.0 * float(a) / float(b)

def parse_quants(txt):
    try:
        vals = [float(x.strip()) for x in txt.split(",")]
        if len(vals) < 5:
            vals = vals + [0.0]*(5-len(vals))
        elif len(vals) > 5:
            vals = vals[:5]
        return vals
    except:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

def format_quants(vals, dec=3):
    if len(vals) != 5:
        vals = (vals + [0.0]*5)[:5]
    fmt = f"{{:.{dec}f}}"
    return "[" + ", ".join(fmt.format(v) for v in vals) + "]"

def make_table(rows, aligns):
    widths = [max(len(strip_ansi(r[i])) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for r in rows:
        cols = []
        for i, c in enumerate(r):
            raw = str(c)
            s = strip_ansi(raw)
            if aligns[i] == "r":
                pad = widths[i] - len(s)
                cols.append(" " * pad + raw)
            else:
                pad = widths[i] - len(s)
                cols.append(raw + " " * pad)
        lines.append(" | ".join(cols))
    sep = "-+-".join("-"*w for w in widths)
    return lines, sep

def normalize_nbatch(nbatch_str: str) -> str:
    s = str(nbatch_str).strip()
    if s.startswith("~"):
        return str(NBATCH_DEFAULT)
    return s

def nbatch_to_int(nbatch_str: str):
    try:
        return int(normalize_nbatch(nbatch_str))
    except Exception:
        return None

def parse_log(path):
    rows = []
    cur_epoch = 0
    qweights = []
    adapts = []
    rls = []
    with open(path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = PAT_BATCH_STATS.search(line)
            if m:
                ep = int(m.group("epoch"))
                cur_epoch = max(cur_epoch, ep)
                b = int(m.group("batch"))
                nb_raw = m.group("nbatch")
                nb_norm = normalize_nbatch(nb_raw)
                kind = m.group("kind")
                bpp_txt = m.group("bpp")
                psnr_txt = m.group("psnr")
                ssim = float(m.group("ssim"))
                msssim = float(m.group("msssim"))
                r = float(m.group("R"))
                rows.append({
                    "epoch": ep,
                    "batch_num": b,
                    "nbatch_raw": nb_raw,
                    "nbatch_norm": nb_norm,
                    "batch_disp": f"{b}/{nb_norm}",
                    "kind": kind,
                    "bpp_txt": f"[{bpp_txt}]",
                    "psnr_txt": f"[{psnr_txt}]",
                    "ssim": ssim,
                    "msssim": msssim,
                    "R": r,
                    "bpp_vals": parse_quants(bpp_txt),
                    "psnr_vals": parse_quants(psnr_txt),
                })
                continue
            mq = PAT_QWEIGHTS.search(line)
            if mq:
                qweights.append({
                    "bpp_q1": float(mq.group("bpp_q1")),
                    "bpp_med": float(mq.group("bpp_med")),
                    "bpp_q3": float(mq.group("bpp_q3")),
                    "psnr_q1": float(mq.group("psnr_q1")),
                    "psnr_med": float(mq.group("psnr_med")),
                    "psnr_q3": float(mq.group("psnr_q3")),
                    "wb": (float(mq.group("wb1")), float(mq.group("wb2")), float(mq.group("wb3"))),
                    "wp": (float(mq.group("wp1")), float(mq.group("wp2")), float(mq.group("wp3"))),
                })
                continue
            ma = PAT_ADAPT.search(line)
            if ma:
                adapts.append({
                    "bpp_tail_ema": float(ma.group("bpp_tail_ema")),
                    "psnr_floor_ema": float(ma.group("psnr_floor_ema")),
                    "req_psnr": float(ma.group("req_psnr")),
                    "req_bpp": float(ma.group("req_bpp")),
                    "d_psnr": float(ma.group("d_psnr")),
                    "d_bpp": float(ma.group("d_bpp")),
                    "alpha_psnr_old": float(ma.group("alpha_psnr_old")),
                    "alpha_psnr_new": float(ma.group("alpha_psnr_new")),
                    "alpha_bpp_old": float(ma.group("alpha_bpp_old")),
                    "alpha_bpp_new": float(ma.group("alpha_bpp_new")),
                })
                continue
            mr = PAT_RL.search(line)
            if mr:
                rls.append({
                    "iter": int(mr.group("iter")),
                    "iters": int(mr.group("iters")),
                    "bpp_min": float(mr.group("bpp_min")),
                    "bpp_q1": float(mr.group("bpp_q1")),
                    "bpp_med": float(mr.group("bpp_med")),
                    "bpp_q3": float(mr.group("bpp_q3")),
                    "bpp_max": float(mr.group("bpp_max")),
                    "psnr_min": float(mr.group("psnr_min")),
                    "psnr_q1": float(mr.group("psnr_q1")),
                    "psnr_med": float(mr.group("psnr_med")),
                    "psnr_q3": float(mr.group("psnr_q3")),
                    "psnr_max": float(mr.group("psnr_max")),
                    "ssim": float(mr.group("ssim")),
                    "msssim": float(mr.group("msssim")),
                    "Wbpp": float(mr.group("Wbpp")),
                    "Wpsnr": float(mr.group("Wpsnr")),
                    "msssim_med": float(mr.group("msssim_med")),
                    "Wtotal": float(mr.group("Wtotal")),
                    "alpha_bpp": float(mr.group("alpha_bpp")),
                    "alpha_psnr": float(mr.group("alpha_psnr")),
                })
                continue
    return rows, cur_epoch, qweights, adapts, rls

def group_by_epoch(rows):
    order = []
    groups = {}
    for r in rows:
        if r["kind"] != "BATCH":
            continue
        ep = r["epoch"]
        if ep not in groups:
            groups[ep] = []
            order.append(ep)
        groups[ep].append(r)
    return order, groups

def epoch_averages(batch_rows):
    n = len(batch_rows)
    if n == 0:
        return {
            "bpp_avg": [0.0]*5,
            "psnr_avg": [0.0]*5,
            "ssim_avg": 0.0,
            "ms_avg": 0.0,
            "r_avg": 0.0,
        }
    bpp_sum = [0.0]*5
    psnr_sum = [0.0]*5
    ssim_sum = 0.0
    ms_sum = 0.0
    r_sum = 0.0
    for r in batch_rows:
        for i in range(5):
            bpp_sum[i] += r["bpp_vals"][i]
            psnr_sum[i] += r["psnr_vals"][i]
        ssim_sum += r["ssim"]
        ms_sum += r["msssim"]
        r_sum += r["R"]
    return {
        "bpp_avg": [x/n for x in bpp_sum],
        "psnr_avg": [x/n for x in psnr_sum],
        "ssim_avg": ssim_sum/n,
        "ms_avg": ms_sum/n,
        "r_avg": r_sum/n,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--highlight", type=float, default=None)
    args = ap.parse_args()

    path = OUT_PATH or newest_log()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not path:
        print(f"[{ts}] No logs found in logs/*.out and OUT_PATH not set.")
        return

    rows, cur_ep, qweights, adapts, rls = parse_log(path)
    print(f"[{ts}] Log: {path}")

    last_row = None
    for r in reversed(rows):
        if r["kind"] == "BATCH" and r["epoch"] == cur_ep:
            last_row = r
            break

    if last_row:
        b_cur = last_row["batch_num"]
        nb_num = nbatch_to_int(last_row["nbatch_norm"])
        if nb_num:
            print(f"Batch progress (epoch {cur_ep}): {b_cur}/{last_row['nbatch_norm']} ({pct(b_cur, nb_num):.2f}%)")
        else:
            print(f"Batch progress (epoch {cur_ep}): {last_row['batch_disp']}")
    else:
        print("Batch progress: (no batches parsed yet)")

    print("-" * 120)

    header = ["Epoch", "Batch", "bpp [min,q1,med,q3,max]", "PSNR [min,q1,med,q3,max]", "SSIM", "MS", "R"]
    aligns = ["r", "r", "l", "l", "r", "r", "r"]
    table = [header]

    epoch_order, groups = group_by_epoch(rows)

    for ep in epoch_order:
        batches = groups[ep]
        avgs = epoch_averages(batches)
        r_avg_ep = avgs["r_avg"]
        for r in batches:
            row = [
                ep,
                r["batch_disp"],
                r["bpp_txt"],
                r["psnr_txt"],
                f"{r['ssim']:.3f}",
                f"{r['msssim']:.3f}",
                f"{r['R']:.3f}",
            ]
            if args.highlight is not None and r["R"] >= args.highlight:
                row = [f"{BLUE}{c}{RESET}" for c in row]
            else:
                color = GREEN if r["R"] > r_avg_ep else (RED if r["R"] < r_avg_ep else "")
                if color:
                    row = [f"{color}{c}{RESET}" for c in row]
            table.append(row)
        total_row = [
            f"{ep} TOTAL",
            f"{len(batches)} rows",
            format_quants(avgs["bpp_avg"], dec=3),
            format_quants(avgs["psnr_avg"], dec=2),
            f"{avgs['ssim_avg']:.3f}",
            f"{avgs['ms_avg']:.3f}",
            f"{avgs['r_avg']:.3f}",
        ]
        table.append(total_row)

    lines, sep = make_table(table, aligns)
    print(lines[0]); print(sep)
    for line in lines[1:]:
        print(line)

    if qweights or adapts or rls:
        print("-" * 120)
    if qweights:
        q = qweights[-1]
        print(f"[QWEIGHTS] bpp_cuts=[{q['bpp_q1']:.3f}, {q['bpp_med']:.3f}, {q['bpp_q3']:.3f}]  "
              f"psnr_cuts=[{q['psnr_q1']:.2f}, {q['psnr_med']:.2f}, {q['psnr_q3']:.2f}]  "
              f"wb={q['wb']}  wp={q['wp']}")
    if adapts:
        a = adapts[-1]
        print(f"[ADAPT] bpp_tail_ema={a['bpp_tail_ema']:.3f}  psnr_floor_ema={a['psnr_floor_ema']:.2f}  "
              f"req_psnr={a['req_psnr']:.2f}  req_bpp={a['req_bpp']:.3f}  "
              f"d_psnr={a['d_psnr']:.2f}  d_bpp={a['d_bpp']:.3f}  "
              f"alpha_psnr:{a['alpha_psnr_old']:.3f}->{a['alpha_psnr_new']:.3f}  "
              f"alpha_bpp:{a['alpha_bpp_old']:.3f}->{a['alpha_bpp_new']:.3f}")
    if rls:
        r = rls[-1]
        print(f"[RL] iter {r['iter']}/{r['iters']}  "
              f"bpp=[{r['bpp_min']:.3f}, {r['bpp_q1']:.3f}, {r['bpp_med']:.3f}, {r['bpp_q3']:.3f}, {r['bpp_max']:.3f}]  "
              f"PSNR=[{r['psnr_min']:.2f}, {r['psnr_q1']:.2f}, {r['psnr_med']:.2f}, {r['psnr_q3']:.2f}, {r['psnr_max']:.2f}]  "
              f"SSIM={r['ssim']:.3f}  MS={r['msssim']:.3f}  Wbpp={r['Wbpp']:.3f}  Wpsnr={r['Wpsnr']:.3f}  "
              f"mSSIM_med={r['msssim_med']:.3f}  Wtotal={r['Wtotal']:.3f}  "
              f"alpha_bpp={r['alpha_bpp']:.3f}  alpha_psnr={r['alpha_psnr']:.3f}")

if __name__ == "__main__":
    main()
