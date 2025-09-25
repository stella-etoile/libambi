#!/usr/bin/env python3
import re, sys, math, argparse, pathlib, statistics

RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"

PAT_BATCH_HDR = re.compile(r'^\[Epoch\s+(?P<epoch>\d+)\/(?P<epochs>\d+)\]\s+Batch\s+(?P<batch>\d+)\/(?P<nbatch>~?\d+)\s*$')
PAT_STATS = re.compile(
    r'^\[Epoch\s+(?P<epoch>\d+)\/(?P<epochs>\d+)\]\s+'
    r'(?P<batch>\d+)\/(?P<nbatch>~?\d+)\s+'
    r'(?P<kind>BATCH|TOTAL)\s+'
    r'bpp=\[(?P<bpp>[^\]]+)\]\s+'
    r'PSNR=\[(?P<psnr>[^\]]+)\]\s+'
    r'MS=(?P<ms>[-+eE\d\.]+)\s+'
    r'R~=(?P<R>[-+eE\d\.]+)\s*$'
)
PAT_TIME = re.compile(
    r'^\[Epoch\s+(?P<epoch>\d+)\/(?P<epochs>\d+)\]\s+'
    r'(?P<batch>\d+)\/(?P<nbatch>~?\d+)\s+TIME\s+'
    r'elapsed=(?P<elapsed>\S+)\s+avg/batch=(?P<avg_batch>\S+)\s+est_epoch=(?P<est_epoch>\S+)\s+eta_epoch=(?P<eta_epoch>\S+)\s*$'
)

def parse_list(s):
    parts = [x.strip() for x in s.split(',')]
    vals = []
    for p in parts:
        if p.endswith('dB'):
            p = p[:-2].strip()
        try:
            vals.append(float(p))
        except:
            vals.append(math.nan)
    while len(vals) < 5:
        vals.append(math.nan)
    return vals[:5]

def format_scalar(val, decimals):
    if math.isnan(val):
        return "nan"
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(val)

def parse_duration_to_seconds(s):
    s = s.strip()
    if s.endswith("ms"):
        try: return float(s[:-2]) / 1000.0
        except: return math.nan
    if s.endswith("s") and "m" not in s and "h" not in s:
        try: return float(s[:-1])
        except: return math.nan
    total = 0.0
    cur = s
    if "h" in cur:
        a, cur = cur.split("h", 1)
        try: total += float(a) * 3600.0
        except: return math.nan
    if "m" in cur:
        a, cur = cur.split("m", 1)
        try: total += float(a) * 60.0
        except: return math.nan
    if cur.endswith("s"):
        cur = cur[:-1]
    if cur:
        try: total += float(cur)
        except: return math.nan
    return total

def seconds_to_hms_str(x):
    if math.isnan(x):
        return "nan"
    x = int(round(x))
    h = x // 3600
    m = (x % 3600) // 60
    s = x % 60
    if h > 0:
        return f"{h}h{m}m{s}s"
    if m > 0:
        return f"{m}m{s}s"
    return f"{s}s"

def color_for_row(r_value, r_avg, blue_mode, blue_threshold, color_mode, use_rrel_for_blue):
    if color_mode == "never":
        return "", ""
    rrel = 0.0 if r_avg == 0 else (r_value - r_avg) / abs(r_avg)
    metric_for_blue = rrel if use_rrel_for_blue else r_value
    if blue_mode and metric_for_blue > blue_threshold:
        return BLUE, RESET
    if r_value > r_avg:
        return GREEN, RESET
    return RED, RESET

def build_list_widths(all_rows_items):
    # all_rows_items: list of list-of-5 strings per row
    widths = [0, 0, 0, 0, 0]
    for items in all_rows_items:
        for i in range(5):
            widths[i] = max(widths[i], len(items[i]))
    return widths

def join_list_with_alignment(items, widths):
    # No space after '['. Left-justify each item to its width, then comma+space.
    return "[" + ", ".join(items[i].ljust(widths[i]) for i in range(5)) + "]"

def prepare_row_strings(rows, total_row):
    # Precompute per-position strings (no leading spaces) for bpp (3dp) and psnr (2dp)
    bpp_items_rows = []
    psnr_items_rows = []
    for r in rows:
        bpp_items_rows.append([format_scalar(v, 3) for v in r["bpp"]])
        psnr_items_rows.append([format_scalar(v, 2) for v in r["psnr"]])
    total_bpp_items = total_psnr_items = None
    if total_row is not None:
        total_bpp_items = [format_scalar(v, 3) for v in total_row["bpp"]]
        total_psnr_items = [format_scalar(v, 2) for v in total_row["psnr"]]
        bpp_items_rows.append(total_bpp_items)
        psnr_items_rows.append(total_psnr_items)
    # compute widths by max length per position
    bpp_widths = build_list_widths(bpp_items_rows) if bpp_items_rows else [0]*5
    psnr_widths = build_list_widths(psnr_items_rows) if psnr_items_rows else [0]*5
    return bpp_widths, psnr_widths, total_bpp_items, total_psnr_items

def build_table_strings(rows, r_avg_epoch, dt_by_batch, eta_str_by_batch,
                        color_mode, blue_mode, blue_threshold, use_rrel_for_blue,
                        total_row, total_dt_avg, total_elapsed_str):
    headers = ["batch", "bpp[min,q1,med,q3,max]", "PSNR[min,q1,med,q3,max]", "MS", "R", "R_rel", "dt_batch", "eta"]

    # Precompute aligned list widths (so commas/bracket align) including TOTAL
    bpp_widths, psnr_widths, total_bpp_items, total_psnr_items = prepare_row_strings(rows, total_row)

    # Build row strings
    raw_rows = []
    for row in rows:
        bpp_items = [format_scalar(v, 3) for v in row["bpp"]]
        psnr_items = [format_scalar(v, 2) for v in row["psnr"]]
        bpp_s = join_list_with_alignment(bpp_items, bpp_widths)
        psnr_s = join_list_with_alignment(psnr_items, psnr_widths)
        r = row["R"]
        rrel = 0.0 if r_avg_epoch == 0 else (r - r_avg_epoch)/abs(r_avg_epoch)
        dt = dt_by_batch.get(row["batch"], math.nan)
        eta_s = eta_str_by_batch.get(row["batch"], "nan")
        raw_rows.append([
            f"{row['batch']}",
            bpp_s,
            psnr_s,
            f"{row['ms']:.3f}",
            f"{r:+.3f}",
            f"{rrel:+.3f}",
            "nan" if math.isnan(dt) else f"{dt:.1f}s",
            eta_s,
        ])

    total_raw = None
    if total_row is not None:
        if total_bpp_items is None:
            total_bpp_items = [format_scalar(v, 3) for v in total_row["bpp"]]
        if total_psnr_items is None:
            total_psnr_items = [format_scalar(v, 2) for v in total_row["psnr"]]
        bpp_s = join_list_with_alignment(total_bpp_items, bpp_widths)
        psnr_s = join_list_with_alignment(total_psnr_items, psnr_widths)
        r = total_row["R"]
        total_raw = [
            "TOTAL",
            bpp_s,
            psnr_s,
            f"{total_row['ms']:.3f}",
            f"{r:+.3f}",
            f"{0.0:+.3f}",
            "nan" if math.isnan(total_dt_avg) else f"{total_dt_avg:.1f}s",
            total_elapsed_str if total_elapsed_str else "nan",
        ]

    # Column widths (use headers + all rows + total)
    widths = []
    all_rows_for_width = list(raw_rows) + ([total_raw] if total_raw else [])
    for i, h in enumerate(headers):
        w = len(h)
        for r in all_rows_for_width:
            w = max(w, len(r[i]))
        widths.append(w)

    lines = []
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "-" * len(header_line)
    lines.append(header_line)
    lines.append(sep)
    for rvals in raw_rows:
        r_value = float(rvals[4])
        c0, c1 = color_for_row(r_value, r_avg_epoch, blue_mode, blue_threshold, color_mode, use_rrel_for_blue)
        out = "  ".join([
            rvals[0].rjust(widths[0]),
            rvals[1].ljust(widths[1]),
            rvals[2].ljust(widths[2]),
            rvals[3].rjust(widths[3]),
            rvals[4].rjust(widths[4]),
            rvals[5].rjust(widths[5]),
            rvals[6].rjust(widths[6]),
            rvals[7].rjust(widths[7]),
        ])
        lines.append(f"{c0}{out}{c1}")
    if total_raw:
        lines.append(sep)
        out = "  ".join([
            total_raw[0].rjust(widths[0]),
            total_raw[1].ljust(widths[1]),
            total_raw[2].ljust(widths[2]),
            total_raw[3].rjust(widths[3]),
            total_raw[4].rjust(widths[4]),
            total_raw[5].rjust(widths[5]),
            total_raw[6].rjust(widths[6]),
            total_raw[7].rjust(widths[7]),
        ])
        lines.append(out)
    lines.append(sep)
    return lines

def print_epoch_table(epoch_id, rows, r_avg_epoch, dt_by_batch, eta_str_by_batch,
                      color_mode, blue_mode, blue_threshold, use_rrel_for_blue,
                      total_row, total_dt_avg, total_elapsed_str):
    if not rows:
        return
    title = f"Epoch {epoch_id}"
    table_lines = build_table_strings(rows, r_avg_epoch, dt_by_batch, eta_str_by_batch,
                                      color_mode, blue_mode, blue_threshold, use_rrel_for_blue,
                                      total_row, total_dt_avg, total_elapsed_str)
    print(title)
    print("-" * len(title))
    for ln in table_lines:
        print(ln)
    print(f"avg_R_total={r_avg_epoch:+.6f}")
    print()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="logs/model.out")
    ap.add_argument("--color", choices=["auto","always","never"], default="auto")
    ap.add_argument("--blue-threshold", type=float, default=None)
    ap.add_argument("--blue-by", choices=["R","Rrel"], default="R")
    args = ap.parse_args()

    p = pathlib.Path(args.file)
    if not p.exists():
        print(f"missing: {p}", file=sys.stderr)
        sys.exit(1)

    current_epoch = None
    epoch_rows = []
    last_total_r_for_epoch = None
    last_total_stats_for_epoch = None
    output_rows_by_epoch = {}
    elapsed_by_epoch_batch = {}
    eta_by_epoch_batch = {}

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            mS = PAT_STATS.match(line)
            if mS:
                ep = int(mS.group("epoch"))
                kind = mS.group("kind")
                batch = int(mS.group("batch"))
                bpp_vals = parse_list(mS.group("bpp"))
                psnr_vals = parse_list(mS.group("psnr"))
                ms = float(mS.group("ms"))
                R = float(mS.group("R"))
                if current_epoch is None:
                    current_epoch = ep
                if ep != current_epoch:
                    r_avg = last_total_r_for_epoch if last_total_r_for_epoch is not None else 0.0
                    output_rows_by_epoch[current_epoch] = (
                        list(epoch_rows),
                        r_avg,
                        dict(elapsed_by_epoch_batch.get(current_epoch, {})),
                        dict(eta_by_epoch_batch.get(current_epoch, {})),
                        last_total_stats_for_epoch
                    )
                    epoch_rows = []
                    last_total_r_for_epoch = None
                    last_total_stats_for_epoch = None
                    current_epoch = ep
                if kind == "TOTAL":
                    last_total_r_for_epoch = R
                    last_total_stats_for_epoch = {
                        "batch": batch,
                        "bpp": bpp_vals,
                        "psnr": psnr_vals,
                        "ms": ms,
                        "R": R,
                    }
                elif kind == "BATCH":
                    epoch_rows.append({
                        "batch": batch,
                        "bpp": bpp_vals,
                        "psnr": psnr_vals,
                        "ms": ms,
                        "R": R,
                    })
                continue
            mT = PAT_TIME.match(line)
            if mT:
                ep = int(mT.group("epoch"))
                batch = int(mT.group("batch"))
                elapsed = parse_duration_to_seconds(mT.group("elapsed"))
                eta_str = mT.group("eta_epoch")
                if ep not in elapsed_by_epoch_batch:
                    elapsed_by_epoch_batch[ep] = {}
                if ep not in eta_by_epoch_batch:
                    eta_by_epoch_batch[ep] = {}
                elapsed_by_epoch_batch[ep][batch] = elapsed
                eta_by_epoch_batch[ep][batch] = eta_str
                continue
            if PAT_BATCH_HDR.match(line):
                continue

    if current_epoch is not None:
        r_avg = last_total_r_for_epoch if last_total_r_for_epoch is not None else 0.0
        output_rows_by_epoch[current_epoch] = (
            list(epoch_rows),
            r_avg,
            dict(elapsed_by_epoch_batch.get(current_epoch, {})),
            dict(eta_by_epoch_batch.get(current_epoch, {})),
            last_total_stats_for_epoch
        )

    if args.color == "auto":
        color_mode = "always" if sys.stdout.isatty() else "never"
    else:
        color_mode = args.color
    blue_mode = args.blue_threshold is not None
    use_rrel_for_blue = args.blue_by == "Rrel"

    for ep in sorted(output_rows_by_epoch.keys()):
        rows, r_avg, elapsed_map, eta_map, total_row = output_rows_by_epoch[ep]
        dt_map = {}
        if elapsed_map:
            for b in sorted(elapsed_map.keys()):
                prev = elapsed_map.get(b-1, None)
                cur = elapsed_map[b]
                if prev is None:
                    dt_map[b] = cur
                else:
                    dt = cur - prev
                    dt_map[b] = dt if dt >= 0 else math.nan
        valid_dts = [v for v in (dt_map.get(b) for b in sorted(dt_map.keys())) if v is not None and not math.isnan(v)]
        total_dt_avg = statistics.mean(valid_dts) if valid_dts else math.nan
        total_elapsed = max(elapsed_map.values()) if elapsed_map else math.nan
        total_elapsed_str = seconds_to_hms_str(total_elapsed) if not math.isnan(total_elapsed) else "nan"

        print_epoch_table(
            ep,
            rows,
            r_avg,
            dt_map,
            eta_map,
            color_mode,
            blue_mode,
            args.blue_threshold if blue_mode else 0.0,
            use_rrel_for_blue,
            total_row,
            total_dt_avg,
            total_elapsed_str
        )

if __name__ == "__main__":
    main()
