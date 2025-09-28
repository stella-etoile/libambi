import re, os, sys, math, argparse, pathlib, statistics, csv

RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"

PAT_BATCH_HDR = re.compile(r'^\[Epoch\s+(?P<epoch>\d+)\/(?P<epochs>\d+)\]\s+Batch\s+(?P<batch>\d+)\/(?P<nbatch>~?\d+)\s*$')
PAT_BATCH_METRICS = re.compile(
    r'^\[Epoch\s+(?P<epoch>\d+)\/(?P<epochs>\d+)\]\s+(?P<batch>\d+)\/(?P<nbatch>~?\d+)\s+(?P<label>BATCH|VAL-BATCH|TOTAL)\s+'
    r'bpp=\[(?P<bpp>[^\]]+)\]\s+PSNR=\[(?P<psnr>[^\]]+)\]\s+MS=(?P<ms>\S+)\s+R~=(?P<R>[-+]?[\d\.eE]+)'
)
PAT_BATCH_ELAPSED = re.compile(
    r'^\[Epoch\s+(?P<epoch>\d+)\/(?P<epochs>\d+)\]\s+(?P<label>TRAIN-BATCH|VAL-BATCH)\s+(?P<batch>\d+)\s+elapsed=(?P<t>[\dhms\.]+)\s*$'
)
PAT_TIME_BLOCK = re.compile(
    r'^\[Epoch\s+(?P<epoch>\d+)\/(?P<epochs>\d+)\]\s+(?P<batch>\d+)\/(?P<nbatch>~?\d+)\s+TIME\s+elapsed=(?P<elapsed>[\dhms\.]+)\s+avg/batch=(?P<avg>[\dhms\.]+)\s+est_epoch=(?P<est>[\dhms\.]+)\s+eta_epoch=(?P<eta>[\dhms\.]+)'
)
PAT_RL_TRAIN = re.compile(
    r'^\[RL\]\s+iter\s+(?P<epoch>\d+)\/(?P<epochs>\d+)\s+'
    r'bpp_med=(?P<bpp_med>[-+]?[\d\.eE]+)\s+\(min=(?P<bpp_min>[-+]?[\d\.eE]+),\s+q1=(?P<bpp_q1>[-+]?[\d\.eE]+),\s+q3=(?P<bpp_q3>[-+]?[\d\.eE]+),\s+max=(?P<bpp_max>[-+]?[\d\.eE]+)\)\s+'
    r'PSNR_med=(?P<psnr_med>[-+]?[\d\.eE]+)\s+dB\s+\(min=(?P<psnr_min>[-+]?[\d\.eE]+),\s+q1=(?P<psnr_q1>[-+]?[\d\.eE]+),\s+q3=(?P<psnr_q3>[-+]?[\d\.eE]+),\s+max=(?P<psnr_max>[-+]?[\d\.eE]+)\)\s+'
    r'MS-SSIM_avg=(?P<ms>[-+]?[\d\.eE]+)\s+Wtotal=(?P<Wtotal>[-+]?[\d\.eE]+)\s+TRAIN_epoch_elapsed=(?P<elapsed>[\dhms\.]+)\s+alpha_bpp=(?P<alpha_bpp>[-+]?[\d\.eE]+)\s+alpha_psnr=(?P<alpha_psnr>[-+]?[\d\.eE]+)'
)
PAT_VAL_SUM = re.compile(
    r'^\[VAL\]\s+iter\s+(?P<epoch>\d+)\/(?P<epochs>\d+)\s+'
    r'bpp_med=(?P<bpp_med>[-+]?[\d\.eE]+)\s+\(min=(?P<bpp_min>[-+]?[\d\.eE]+),\s+q1=(?P<bpp_q1>[-+]?[\d\.eE]+),\s+q3=(?P<bpp_q3>[-+]?[\d\.eE]+),\s+max=(?P<bpp_max>[-+]?[\d\.eE]+)\)\s+'
    r'PSNR_med=(?P<psnr_med>[-+]?[\d\.eE]+)\s+dB\s+\(min=(?P<psnr_min>[-+]?[\d\.eE]+),\s+q1=(?P<psnr_q1>[-+]?[\d\.eE]+),\s+q3=(?P<psnr_q3>[-+]?[\d\.eE]+),\s+max=(?P<psnr_max>[-+]?[\d\.eE]+)\)\s+'
    r'MS-SSIM_avg=(?P<ms>[-+]?[\d\.eE]+)\s+Wtotal=(?P<Wtotal>[-+]?[\d\.eE]+)\s+VAL_epoch_elapsed=(?P<elapsed>[\dhms\.]+)'
)
PAT_EARLY = re.compile(
    r'^\[RL\]\s+\(val-based\)\s+improved:\s+val_bpp_med=(?P<imp_bpp>\w+),\s+val_psnr_med=(?P<imp_psnr>\w+),\s+wait=(?P<wait>\d+)\/(?P<patience>\d+)'
)
PAT_SAVE = re.compile(r'^\[RL\]\s+saved epoch checkpoint -> (?P<path>.+)$')
PAT_VAL_PROGRESS = re.compile(r'^\[Epoch\s+(?P<epoch>\d+)\/(?P<epochs>\d+)\]\s+VAL\s+(?P<k>\d+)\/(?P<n>~?\d+)')
PAT_PROGRESS = re.compile(
    r'^\[PROGRESS\]\s+(?P<label>.+?)\s+(?P<pct>\d+)%\s+\((?P<done>\d+)\/(?P<total>\d+)\)\s+elapsed=(?P<elapsed>[\d\.]+)s'
)

def parse_time_to_seconds(s):
    s = s.strip()
    if s.endswith('ms'):
        try: return float(s[:-2]) / 1000.0
        except: return math.nan
    if s.endswith('s') and 'm' not in s and 'h' not in s:
        try: return float(s[:-1])
        except: return math.nan
    h = m = sec = 0.0
    mh = re.search(r'(?P<h>\d+)h', s)
    if mh: h = float(mh.group('h'))
    mm = re.search(r'(?P<m>\d+)m', s)
    if mm: m = float(mm.group('m'))
    ms = re.search(r'(?P<s>[\d\.]+)s', s)
    if ms: sec = float(ms.group('s'))
    return h*3600 + m*60 + sec

def parse_list5(x):
    parts = [p.strip() for p in x.split(',')]
    out = []
    for p in parts:
        if p.endswith('dB'): p = p[:-2].strip()
        try: out.append(float(p))
        except: out.append(math.nan)
    while len(out) < 5: out.append(math.nan)
    return out[:5]

def to_bool(x):
    return str(x).lower() == 'true'

def write_csv(path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in header})

def parse_files_to_csv(files, outdir):
    per_batch = []
    per_epoch = {}
    last_checkpoint = {}
    per_progress = []
    cur_epoch = None
    cur_phase = None
    cur_batch = None
    for filepath in files:
        with open(filepath, 'r', errors='ignore') as f:
            for line in f:
                line = line.rstrip('\n')
                m = PAT_BATCH_HDR.match(line)
                if m:
                    cur_epoch = int(m.group('epoch'))
                    cur_phase = 'train'
                    cur_batch = int(m.group('batch'))
                    continue
                m = PAT_VAL_PROGRESS.match(line)
                if m:
                    cur_epoch = int(m.group('epoch'))
                    cur_phase = 'val'
                    cur_batch = int(m.group('k'))
                    continue
                m = PAT_PROGRESS.match(line)
                if m and cur_epoch is not None and cur_phase is not None and cur_batch is not None:
                    per_progress.append({
                        'file': os.path.basename(filepath),
                        'epoch': cur_epoch,
                        'phase': cur_phase,
                        'batch': cur_batch,
                        'label': m.group('label'),
                        'pct': int(m.group('pct')),
                        'done': int(m.group('done')),
                        'total': int(m.group('total')),
                        'elapsed_s': float(m.group('elapsed')),
                    })
                    continue
                m = PAT_BATCH_METRICS.match(line)
                if m:
                    epoch = int(m.group('epoch'))
                    epochs_total = int(m.group('epochs'))
                    batch = int(m.group('batch'))
                    nbatch = m.group('nbatch')
                    label = m.group('label')
                    bpp = parse_list5(m.group('bpp'))
                    psnr = parse_list5(m.group('psnr'))
                    ms = float(m.group('ms'))
                    R = float(m.group('R'))
                    phase = 'train' if label in ('BATCH','TOTAL') else 'val'
                    per_batch.append({
                        'file': os.path.basename(filepath),
                        'epoch': epoch,
                        'epochs_total': epochs_total,
                        'phase': phase,
                        'batch': batch,
                        'nbatch': nbatch,
                        'bpp_min': bpp[0], 'bpp_q1': bpp[1], 'bpp_med': bpp[2], 'bpp_q3': bpp[3], 'bpp_max': bpp[4],
                        'psnr_min': psnr[0], 'psnr_q1': psnr[1], 'psnr_med': psnr[2], 'psnr_q3': psnr[3], 'psnr_max': psnr[4],
                        'ms': ms,
                        'R': R,
                        'batch_elapsed_s': ''
                    })
                    continue
                m = PAT_BATCH_ELAPSED.match(line)
                if m:
                    epoch = int(m.group('epoch'))
                    label = m.group('label')
                    batch = int(m.group('batch'))
                    t = parse_time_to_seconds(m.group('t'))
                    phase = 'train' if label == 'TRAIN-BATCH' else 'val'
                    for i in range(len(per_batch)-1, -1, -1):
                        r = per_batch[i]
                        if r['epoch'] == epoch and r['phase'] == phase and r['batch'] == batch and (r['batch_elapsed_s'] in ('', None)):
                            r['batch_elapsed_s'] = t
                            break
                    continue
                m = PAT_TIME_BLOCK.match(line)
                if m:
                    continue
                m = PAT_RL_TRAIN.match(line)
                if m:
                    epoch = int(m.group('epoch'))
                    per_epoch.setdefault(epoch, {})
                    per_epoch[epoch].update({
                        'epoch': epoch,
                        'epochs_total': int(m.group('epochs')),
                        'train_bpp_med': float(m.group('bpp_med')),
                        'train_bpp_min': float(m.group('bpp_min')),
                        'train_bpp_q1': float(m.group('bpp_q1')),
                        'train_bpp_q3': float(m.group('bpp_q3')),
                        'train_bpp_max': float(m.group('bpp_max')),
                        'train_psnr_med': float(m.group('psnr_med')),
                        'train_psnr_min': float(m.group('psnr_min')),
                        'train_psnr_q1': float(m.group('psnr_q1')),
                        'train_psnr_q3': float(m.group('psnr_q3')),
                        'train_psnr_max': float(m.group('psnr_max')),
                        'train_ms_avg': float(m.group('ms')),
                        'train_Wtotal': float(m.group('Wtotal')),
                        'train_epoch_elapsed_s': parse_time_to_seconds(m.group('elapsed')),
                        'alpha_bpp': float(m.group('alpha_bpp')),
                        'alpha_psnr': float(m.group('alpha_psnr')),
                    })
                    continue
                m = PAT_VAL_SUM.match(line)
                if m:
                    epoch = int(m.group('epoch'))
                    per_epoch.setdefault(epoch, {})
                    per_epoch[epoch].update({
                        'val_bpp_med': float(m.group('bpp_med')),
                        'val_bpp_min': float(m.group('bpp_min')),
                        'val_bpp_q1': float(m.group('bpp_q1')),
                        'val_bpp_q3': float(m.group('bpp_q3')),
                        'val_bpp_max': float(m.group('bpp_max')),
                        'val_psnr_med': float(m.group('psnr_med')),
                        'val_psnr_min': float(m.group('psnr_min')),
                        'val_psnr_q1': float(m.group('psnr_q1')),
                        'val_psnr_q3': float(m.group('psnr_q3')),
                        'val_psnr_max': float(m.group('psnr_max')),
                        'val_ms_avg': float(m.group('ms')),
                        'val_Wtotal': float(m.group('Wtotal')),
                        'val_epoch_elapsed_s': parse_time_to_seconds(m.group('elapsed')),
                    })
                    continue
                m = PAT_EARLY.match(line)
                if m:
                    epoch = None
                    if per_batch:
                        epoch = per_batch[-1]['epoch']
                    elif per_epoch:
                        epoch = sorted(per_epoch.keys())[-1]
                    if epoch is not None:
                        per_epoch.setdefault(epoch, {})
                        per_epoch[epoch].update({
                            'improved_bpp': str(m.group('imp_bpp')).lower() == 'true',
                            'improved_psnr': str(m.group('imp_psnr')).lower() == 'true',
                            'early_wait': int(m.group('wait')),
                            'early_patience': int(m.group('patience')),
                        })
                    continue
                m = PAT_SAVE.match(line)
                if m:
                    epoch = None
                    if per_batch:
                        epoch = per_batch[-1]['epoch']
                    elif per_epoch:
                        epoch = sorted(per_epoch.keys())[-1]
                    if epoch is not None:
                        last_checkpoint[epoch] = m.group('path').strip()
                    continue
                m = PAT_VAL_PROGRESS.match(line)
                if m:
                    continue
    for e, p in per_epoch.items():
        if e in last_checkpoint:
            p['checkpoint'] = last_checkpoint[e]
    batch_header = [
        'file','epoch','epochs_total','phase','batch','nbatch',
        'bpp_min','bpp_q1','bpp_med','bpp_q3','bpp_max',
        'psnr_min','psnr_q1','psnr_med','psnr_q3','psnr_max',
        'ms','R','batch_elapsed_s'
    ]
    epoch_header = [
        'epoch','epochs_total',
        'train_bpp_min','train_bpp_q1','train_bpp_med','train_bpp_q3','train_bpp_max',
        'train_psnr_min','train_psnr_q1','train_psnr_med','train_psnr_q3','train_psnr_max',
        'train_ms_avg','train_Wtotal','train_epoch_elapsed_s','alpha_bpp','alpha_psnr',
        'val_bpp_min','val_bpp_q1','val_bpp_med','val_bpp_q3','val_bpp_max',
        'val_psnr_min','val_psnr_q1','val_psnr_med','val_psnr_q3','val_psnr_max',
        'val_ms_avg','val_Wtotal','val_epoch_elapsed_s',
        'improved_bpp','improved_psnr','early_wait','early_patience','checkpoint'
    ]
    progress_header = ['file','epoch','phase','batch','label','pct','done','total','elapsed_s']
    outdir = pathlib.Path(outdir)
    write_csv(outdir/'per_batch.csv', per_batch, batch_header)
    rows = [per_epoch[k] for k in sorted(per_epoch.keys())]
    write_csv(outdir/'per_epoch.csv', rows, epoch_header)
    write_csv(outdir/'per_progress.csv', per_progress, progress_header)
    return str(outdir/'per_batch.csv'), str(outdir/'per_epoch.csv'), str(outdir/'per_progress.csv')

def to_float(x):
    try:
        if x is None or x == '':
            return math.nan
        return float(x)
    except:
        return math.nan

def format_scalar(val, decimals):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "nan"
    return f"{val:.{decimals}f}"

def build_list_widths(all_rows_items):
    widths = [0, 0, 0, 0, 0]
    for items in all_rows_items:
        for i in range(5):
            widths[i] = max(widths[i], len(items[i]))
    return widths

def join_list_with_alignment(items, widths):
    return "[" + ", ".join(items[i].ljust(widths[i]) for i in range(5)) + "]"

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

def seconds_to_hms_str(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
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

def build_rows_from_csv(per_batch_csv, per_epoch_csv, per_progress_csv, phase):
    rows_by_epoch = {}
    r_avg_by_epoch = {}
    total_row_by_epoch = {}
    dt_by_epoch_batch = {}
    prog_by_epoch_batch = {}
    with open(per_batch_csv, "r", encoding="utf-8", errors="ignore") as fb:
        r = csv.DictReader(fb)
        for d in r:
            if d.get("phase","").strip().lower() != phase:
                continue
            ep = int(d["epoch"])
            batch = int(d["batch"])
            bpp = [to_float(d.get(k)) for k in ["bpp_min","bpp_q1","bpp_med","bpp_q3","bpp_max"]]
            psnr = [to_float(d.get(k)) for k in ["psnr_min","psnr_q1","psnr_med","psnr_q3","psnr_max"]]
            ms = to_float(d.get("ms"))
            R = to_float(d.get("R"))
            rows_by_epoch.setdefault(ep, [])
            rows_by_epoch[ep].append({"batch": batch, "bpp": bpp, "psnr": psnr, "ms": ms, "R": R})
            dt = to_float(d.get("batch_elapsed_s"))
            dt_by_epoch_batch.setdefault(ep, {})
            dt_by_epoch_batch[ep][batch] = dt if not math.isnan(dt) else None
    with open(per_epoch_csv, "r", encoding="utf-8", errors="ignore") as fe:
        r = csv.DictReader(fe)
        for d in r:
            ep = int(d["epoch"])
            if phase == "train":
                bpp = [to_float(d.get(k)) for k in ["train_bpp_min","train_bpp_q1","train_bpp_med","train_bpp_q3","train_bpp_max"]]
                psnr = [to_float(d.get(k)) for k in ["train_psnr_min","train_psnr_q1","train_psnr_med","train_psnr_q3","train_psnr_max"]]
                ms = to_float(d.get("train_ms_avg"))
                R = to_float(d.get("train_Wtotal"))
                r_avg_by_epoch[ep] = 0.0 if math.isnan(R) else R
                total_row_by_epoch[ep] = {"batch": 0, "bpp": bpp, "psnr": psnr, "ms": ms, "R": R}
            else:
                bpp = [to_float(d.get(k)) for k in ["val_bpp_min","val_bpp_q1","val_bpp_med","val_bpp_q3","val_bpp_max"]]
                psnr = [to_float(d.get(k)) for k in ["val_psnr_min","val_psnr_q1","val_psnr_med","val_psnr_q3","val_psnr_max"]]
                ms = to_float(d.get("val_ms_avg"))
                R = to_float(d.get("val_Wtotal"))
                r_avg_by_epoch[ep] = 0.0 if math.isnan(R) else R
                total_row_by_epoch[ep] = {"batch": 0, "bpp": bpp, "psnr": psnr, "ms": ms, "R": R}
    with open(per_progress_csv, "r", encoding="utf-8", errors="ignore") as fp:
        r = csv.DictReader(fp)
        for d in r:
            if d.get("phase","").strip().lower() != phase:
                continue
            ep = int(d["epoch"])
            batch = int(d["batch"])
            label = d["label"]
            pct = int(d["pct"])
            done = int(d["done"])
            total = int(d["total"])
            elapsed_s = to_float(d["elapsed_s"])
            prog_by_epoch_batch.setdefault(ep, {})
            prog_by_epoch_batch[ep].setdefault(batch, {})
            prev = prog_by_epoch_batch[ep][batch].get(label)
            if (prev is None) or (pct > prev["pct"]) or (pct == prev["pct"] and (elapsed_s or -1) >= (prev["elapsed_s"] or -1)):
                prog_by_epoch_batch[ep][batch][label] = {
                    "label": label,
                    "pct": pct,
                    "done": done,
                    "total": total,
                    "elapsed_s": elapsed_s,
                }
    for ep, rows in rows_by_epoch.items():
        rows.sort(key=lambda x: x["batch"])
    return rows_by_epoch, r_avg_by_epoch, total_row_by_epoch, dt_by_epoch_batch, prog_by_epoch_batch

def compute_total_from_batches(rows):
    if not rows:
        return None, 0.0
    bpp_cols = list(zip(*[r["bpp"] for r in rows]))
    psnr_cols = list(zip(*[r["psnr"] for r in rows]))
    bpp_avg = [statistics.fmean([v for v in col if not math.isnan(v)]) if any(not math.isnan(v) for v in col) else math.nan for col in bpp_cols]
    psnr_avg = [statistics.fmean([v for v in col if not math.isnan(v)]) if any(not math.isnan(v) for v in col) else math.nan for col in psnr_cols]
    ms_avg = statistics.fmean([r["ms"] for r in rows if not math.isnan(r["ms"])]) if any(not math.isnan(r["ms"]) for r in rows) else math.nan
    r_avg = statistics.fmean([r["R"] for r in rows if not math.isnan(r["R"])]) if any(not math.isnan(r["R"]) for r in rows) else 0.0
    return {"batch": 0, "bpp": bpp_avg, "psnr": psnr_avg, "ms": ms_avg, "R": r_avg}, r_avg

def prepare_row_strings(rows, total_row):
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
    bpp_widths = build_list_widths(bpp_items_rows) if bpp_items_rows else [0]*5
    psnr_widths = build_list_widths(psnr_items_rows) if psnr_items_rows else [0]*5
    return bpp_widths, psnr_widths, total_bpp_items, total_psnr_items

def build_table_strings(rows, r_avg_epoch, dt_by_batch, prog_by_batch,
                        color_mode, blue_mode, blue_threshold, use_rrel_for_blue,
                        total_row, total_dt_avg, total_elapsed_str):
    headers = ["batch", "bpp[min,q1,med,q3,max]", "PSNR[min,q1,med,q3,max]", "MS", "R", "R_rel", "dt_batch", "eta"]
    bpp_widths, psnr_widths, total_bpp_items, total_psnr_items = prepare_row_strings(rows, total_row)
    raw_rows = []
    for row in rows:
        bpp_items = [format_scalar(v, 3) for v in row["bpp"]]
        psnr_items = [format_scalar(v, 2) for v in row["psnr"]]
        bpp_s = join_list_with_alignment(bpp_items, bpp_widths)
        psnr_s = join_list_with_alignment(psnr_items, psnr_widths)
        r = row["R"]
        rrel = 0.0 if r_avg_epoch == 0 else (r - r_avg_epoch)/abs(r_avg_epoch)
        dt = dt_by_batch.get(row["batch"], math.nan)
        raw_rows.append([
            f"{row['batch']}",
            bpp_s,
            psnr_s,
            f"{row['ms']:.3f}",
            f"{r:+.3f}",
            f"{rrel:+.3f}",
            "nan" if (dt is None or math.isnan(dt)) else f"{dt:.1f}s",
            "nan",
        ])
        plist_map = prog_by_batch.get(row["batch"], {})
        for lbl in sorted(plist_map.keys()):
            p = plist_map[lbl]
            raw_rows.append([
                "PROG",
                p["label"],
                f"{int(p['pct'])}% ({p['done']}/{p['total']})",
                "-",
                "-",
                "-",
                "nan" if (p["elapsed_s"] is None or math.isnan(p["elapsed_s"])) else f"{p['elapsed_s']:.1f}s",
                "-",
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
            "nan" if (total_dt_avg is None or math.isnan(total_dt_avg)) else f"{total_dt_avg:.1f}s",
            total_elapsed_str if total_elapsed_str else "nan",
        ]
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
        if rvals[0] != "PROG":
            r_value = float(rvals[4])
            c0, c1 = color_for_row(
                r_value, r_avg_epoch,
                blue_mode, blue_threshold,
                color_mode, use_rrel_for_blue
            )
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
        else:
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
            lines.append(out)
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

def print_epoch_table(title, epoch_id, rows, r_avg_epoch, dt_by_batch, prog_by_batch,
                      color_mode, blue_mode, blue_threshold, use_rrel_for_blue,
                      total_row, total_dt_avg, total_elapsed_str):
    if not rows:
        return
    t = f"{title} — Epoch {epoch_id}"
    print(t)
    print("-" * len(t))
    for ln in build_table_strings(
        rows, r_avg_epoch, dt_by_batch, prog_by_batch,
        color_mode, blue_mode, blue_threshold, use_rrel_for_blue,
        total_row, total_dt_avg, total_elapsed_str
    ):
        print(ln)
    print()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--out", default="logs")
    ap.add_argument("--color", choices=["auto","always","never"], default="auto")
    ap.add_argument("--blue-threshold", type=float, default=None)
    ap.add_argument("--blue-by", choices=["R","Rrel"], default="R")
    args = ap.parse_args()

    per_batch_csv, per_epoch_csv, per_progress_csv = parse_files_to_csv([args.file], args.out)

    if args.color == "auto":
        color_mode = "always" if sys.stdout.isatty() else "never"
    else:
        color_mode = args.color
    blue_mode = args.blue_threshold is not None
    use_rrel_for_blue = (args.blue_by == "Rrel")
    blue_threshold = args.blue_threshold if blue_mode else 0.0

    rows_train, ravg_train, total_train, dt_train, prog_train = build_rows_from_csv(per_batch_csv, per_epoch_csv, per_progress_csv, "train")
    rows_val, ravg_val, total_val, dt_val, prog_val = build_rows_from_csv(per_batch_csv, per_epoch_csv, per_progress_csv, "val")

    def compute_totals(rows, dt_map):
        dts = []
        out_dt = {}
        for r in rows:
            dt = dt_map.get(r["batch"], None)
            out_dt[r["batch"]] = dt if dt is not None else math.nan
            if dt is not None:
                dts.append(dt)
        avg_dt = statistics.fmean(dts) if dts else math.nan
        total_elapsed = sum(dts) if dts else math.nan
        return out_dt, avg_dt, seconds_to_hms_str(total_elapsed)

    epochs = sorted(set(rows_train.keys()) | set(rows_val.keys()))
    for ep in epochs:
        if ep in rows_train:
            rows = rows_train[ep]
            comp_total, ravg_from_batches = compute_total_from_batches(rows)
            r_avg = ravg_train.get(ep, 0.0) or ravg_from_batches
            dt_map, total_dt_avg, total_elapsed_str = compute_totals(rows, dt_train.get(ep, {}))
            prog_map = prog_train.get(ep, {})
            total_row = comp_total if comp_total is not None else total_train.get(ep, None)
            print_epoch_table("TRAIN", ep, rows, r_avg, dt_map, prog_map,
                              color_mode, blue_mode, blue_threshold, use_rrel_for_blue,
                              total_row, total_dt_avg, total_elapsed_str)
        if ep in rows_val:
            rows = rows_val[ep]
            comp_total, ravg_from_batches = compute_total_from_batches(rows)
            r_avg = ravg_val.get(ep, 0.0) or ravg_from_batches
            dt_map, total_dt_avg, total_elapsed_str = compute_totals(rows, dt_val.get(ep, {}))
            prog_map = prog_val.get(ep, {})
            total_row = comp_total if comp_total is not None else total_val.get(ep, None)
            print_epoch_table("VAL", ep, rows, r_avg, dt_map, prog_map,
                              color_mode, blue_mode, blue_threshold, use_rrel_for_blue,
                              total_row, total_dt_avg, total_elapsed_str)

if __name__ == "__main__":
    main()
