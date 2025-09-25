# plot_ambi_logs.py
import re, math, io, os, sys, gzip
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps as mpl_cmaps

ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

PAT1 = re.compile(
    r'^\s*\[Epoch\s+(?P<epoch>\d+)\s*/\s*(?P<epochs>\d+)\]\s+'
    r'(?P<batch>\d+)\s*/\s*(?P<nbatch>~?\d+)\s+'
    r'(?P<kind>BATCH|TOTAL)\s+'
    r'bpp=\[(?P<bpp>[^\]]+)\]\s+'
    r'PSNR=\[(?P<psnr>[^\]]+)\]\s+'
    r'(?:SSIM=(?P<ssim>[-+eE\d\.NaInf]+)\s+)?'
    r'(?:MS=(?P<ms>[-+eE\d\.NaInf]+)\s+)?'
    r'R(?:~)?=(?P<r>[-+eE\d\.NaInf]+)\s*$'
)

PAT2 = re.compile(
    r'^\s*(?P<epoch>\d+)\s*\|\s*(?P<batch>\d+)\s*/\s*(?P<nbatch>~?\d+)\s*\|\s*'
    r'\[(?P<bpp>[^\]]+)\]\s*\|\s*\[(?P<psnr>[^\]]+)\]\s*\|\s*'
    r'(?P<ssim>[-+eE\d\.NaInf]+)\s*\|\s*(?P<ms>[-+eE\d\.NaInf]+)\s*\|\s*(?P<r>[-+eE\d\.NaInf]+)\s*$'
)

def strip_ansi(s):
    return ANSI_RE.sub('', s)

def _to_float(x):
    try:
        if x is None: return np.nan
        if x.lower() in ('nan','na','inf','-inf','infinity','-infinity'): return float(x)
        return float(x)
    except:
        return np.nan

def parse_five(x):
    try:
        xs = [t.strip() for t in x.split(',')]
        vals = [_to_float(t) for t in xs]
        if len(vals) == 5: return vals
        if len(vals) > 5: return vals[:5]
    except:
        pass
    return [np.nan]*5

def make_row(m, kind_default):
    epoch = int(m.group('epoch'))
    kind = m.groupdict().get('kind') or kind_default
    batch = int(m.group('batch'))
    bpp = parse_five(m.group('bpp'))
    psnr = parse_five(m.group('psnr'))
    ssim = _to_float(m.groupdict().get('ssim'))
    ms = _to_float(m.groupdict().get('ms'))
    r = _to_float(m.groupdict().get('r'))
    return dict(
        epoch=epoch, batch=batch, kind=kind,
        bpp_min=bpp[0], bpp_q1=bpp[1], bpp_med=bpp[2], bpp_q3=bpp[3], bpp_max=bpp[4],
        psnr_min=psnr[0], psnr_q1=psnr[1], psnr_med=psnr[2], psnr_q3=psnr[3], psnr_max=psnr[4],
        ssim=ssim, ms=ms, reward=r
    )

def _iter_lines(path):
    if not path.exists():
        return
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                yield line
    else:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                yield line

def parse_log_multi(paths):
    rows = []
    n1 = n2 = 0
    verbose = os.environ.get("VERBOSE","0") == "1"
    for p in paths:
        for line in _iter_lines(p):
            s = strip_ansi(line).replace('’','\'').strip()
            if not s: continue
            m1 = PAT1.match(s)
            if m1:
                rows.append(make_row(m1, None)); n1 += 1
                continue
            m2 = PAT2.match(s)
            if m2:
                rows.append(make_row(m2, "BATCH")); n2 += 1
                continue
            if 'bpp=[' in s and 'PSNR=[' in s and 'R' in s:
                s2 = re.sub(r'\s+', ' ', s)
                m1b = PAT1.match(s2)
                if m1b:
                    rows.append(make_row(m1b, None)); n1 += 1
                    continue
    if verbose:
        print(f"matched PAT1={n1} PAT2={n2} total={len(rows)}")
    if not rows:
        return pd.DataFrame(columns=[
            'epoch','batch','kind','bpp_min','bpp_q1','bpp_med','bpp_q3','bpp_max',
            'psnr_min','psnr_q1','psnr_med','psnr_q3','psnr_max','ssim','ms','reward'
        ])
    df = pd.DataFrame(rows)
    df = df.sort_values(['epoch','batch','kind']).reset_index(drop=True)
    return df

def psnr(img1, img2):
    a = np.asarray(img1).astype(np.float32)
    b = np.asarray(img2).astype(np.float32)
    mse = np.mean((a-b)**2)
    if mse == 0:
        return 100.0
    return 10*math.log10(255.0**2/mse)

def list_images(dirpath):
    p = Path(dirpath)
    exts = ['*.png','*.jpg','*.jpeg','*.bmp']
    files = []
    for e in exts:
        files.extend(p.glob(e))
    return sorted(files)

def jpeg_rd_per_image(img_dir, qualities):
    files = list_images(img_dir)
    rows = []
    for q in qualities:
        for f in files:
            img = Image.open(f).convert('RGB')
            w,h = img.size
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=q, subsampling=0, optimize=True)
            size = len(buf.getvalue())
            bpp = (size*8)/(w*h)
            buf.seek(0)
            rec = Image.open(buf)
            rows.append(dict(file=str(f.name), quality=q, bpp=bpp, psnr=psnr(img, rec)))
    return pd.DataFrame(rows)

def jpeg_rd_mean(df):
    g = df.groupby('quality').agg(bpp_mean=('bpp','mean'),
                                  bpp_q1=('bpp',lambda x: np.nanquantile(x,0.25)),
                                  bpp_med=('bpp','median'),
                                  bpp_q3=('bpp',lambda x: np.nanquantile(x,0.75)),
                                  psnr_mean=('psnr','mean'),
                                  psnr_q1=('psnr',lambda x: np.nanquantile(x,0.25)),
                                  psnr_med=('psnr','median'),
                                  psnr_q3=('psnr',lambda x: np.nanquantile(x,0.75)),
                                  n=('psnr','count')).reset_index()
    return g

def _norm_colors(values, cmap_name='viridis'):
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(v)
    if not mask.any():
        return ['#808080']*len(v), None, None
    vmin = np.nanmin(v[mask]); vmax = np.nanmax(v[mask])
    if vmax == vmin: vmax = vmin + 1e-9
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = mpl_cmaps.get_cmap(cmap_name)
    cols = [mcolors.to_hex(cmap(norm(x))) if np.isfinite(x) else '#808080' for x in v]
    return cols, norm, cmap

def plot_rd_with_jpeg(df_model, kind, out_prefix, jpeg_mean=None):
    d = df_model[df_model['kind']==kind].sort_values(['epoch','batch'])
    if d.empty: return
    fig = plt.figure(figsize=(7,5), dpi=140); ax = plt.gca()
    ax.plot(d['bpp_med'], d['psnr_med'], marker='o', label='Model (med)')
    ax.plot(d['bpp_q1'], d['psnr_q1'], marker='.', linestyle='--', label='Model (q1)')
    ax.plot(d['bpp_q3'], d['psnr_q3'], marker='.', linestyle='--', label='Model (q3)')
    if jpeg_mean is not None and not jpeg_mean.empty:
        ax.plot(jpeg_mean['bpp_mean'], jpeg_mean['psnr_mean'], marker='x', label='JPEG')
        if 'psnr_q1' in jpeg_mean and 'psnr_q3' in jpeg_mean:
            ax.plot(jpeg_mean['bpp_mean'], jpeg_mean['psnr_q1'], linestyle=':', label='JPEG (q1)')
            ax.plot(jpeg_mean['bpp_mean'], jpeg_mean['psnr_q3'], linestyle=':', label='JPEG (q3)')
    ax.set_xlabel('bpp'); ax.set_ylabel('PSNR (dB)'); ax.set_title(f'RD curve [{kind}] (linear)')
    ax.grid(True, linestyle='--', alpha=0.4); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out_prefix}_linear_with_jpeg.png"); plt.close(fig)

    mask_med = d['bpp_med'].values > 0
    mask_q1 = d['bpp_q1'].values > 0
    mask_q3 = d['bpp_q3'].values > 0
    fig = plt.figure(figsize=(7,5), dpi=140); ax = plt.gca()
    if mask_med.any():
        ax.plot(np.log2(d['bpp_med'][mask_med]), d['psnr_med'][mask_med], marker='o', label='Model (med)')
    if mask_q1.any():
        ax.plot(np.log2(d['bpp_q1'][mask_q1]), d['psnr_q1'][mask_q1], marker='.', linestyle='--', label='Model (q1)')
    if mask_q3.any():
        ax.plot(np.log2(d['bpp_q3'][mask_q3]), d['psnr_q3'][mask_q3], marker='.', linestyle='--', label='Model (q3)')
    if jpeg_mean is not None and not jpeg_mean.empty:
        jm = jpeg_mean[jpeg_mean['bpp_mean']>0]
        if not jm.empty:
            ax.plot(np.log2(jm['bpp_mean']), jm['psnr_mean'], marker='x', label='JPEG')
            if 'psnr_q1' in jm and 'psnr_q3' in jm:
                ax.plot(np.log2(jm['bpp_mean']), jm['psnr_q1'], linestyle=':', label='JPEG (q1)')
                ax.plot(np.log2(jm['bpp_mean']), jm['psnr_q3'], linestyle=':', label='JPEG (q3)')
    ax.set_xlabel('log2(bpp)'); ax.set_ylabel('PSNR (dB)'); ax.set_title(f'RD curve [{kind}] (log2 bpp)')
    ax.grid(True, linestyle='--', alpha=0.4); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out_prefix}_log2_with_jpeg.png"); plt.close(fig)

def plot_rd_with_labels(df_model, kind, out_prefix, jpeg_mean=None):
    d = df_model[df_model['kind']==kind].sort_values(['epoch','batch'])
    if d.empty: return
    labels = [f"{e}-{b}" for e,b in zip(d['epoch'].values, d['batch'].values)]
    cols, norm, cmap = _norm_colors(d['reward'].values)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap) if norm is not None else None
    if sm is not None: sm.set_array([])
    fig = plt.figure(figsize=(7,5), dpi=140); ax = plt.gca()
    ax.plot(d['bpp_med'], d['psnr_med'], marker='o', label='Model (med)')
    for xi, yi, lab, c in zip(d['bpp_med'].values, d['psnr_med'].values, labels, cols):
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(5,5), fontsize=6, color=c)
    ax.plot(d['bpp_q1'], d['psnr_q1'], marker='.', linestyle='--', label='Model (q1)')
    ax.plot(d['bpp_q3'], d['psnr_q3'], marker='.', linestyle='--', label='Model (q3)')
    if jpeg_mean is not None and not jpeg_mean.empty:
        ax.plot(jpeg_mean['bpp_mean'], jpeg_mean['psnr_mean'], marker='x', label='JPEG')
    if sm is not None:
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04); cbar.set_label('Normalized reward', rotation=270, labelpad=12)
    ax.set_xlabel('bpp'); ax.set_ylabel('PSNR (dB)'); ax.set_title(f'RD curve [{kind}] with labels (linear)')
    ax.grid(True, linestyle='--', alpha=0.4); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out_prefix}_linear_labels.png"); plt.close(fig)

    mask_med = d['bpp_med'].values > 0
    fig = plt.figure(figsize=(7,5), dpi=140); ax = plt.gca()
    if mask_med.any():
        lx = np.log2(d['bpp_med'][mask_med]); ly = d['psnr_med'][mask_med]
        llabels = list(np.array(labels)[mask_med]); lcols = list(np.array(cols)[mask_med])
        ax.plot(lx, ly, marker='o', label='Model (med)')
        for xi, yi, lab, c in zip(lx, ly, llabels, lcols):
            ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(5,5), fontsize=6, color=c)
    mask_q1 = d['bpp_q1'].values > 0; mask_q3 = d['bpp_q3'].values > 0
    if mask_q1.any():
        ax.plot(np.log2(d['bpp_q1'][mask_q1]), d['psnr_q1'][mask_q1], marker='.', linestyle='--', label='Model (q1)')
    if mask_q3.any():
        ax.plot(np.log2(d['bpp_q3'][mask_q3]), d['psnr_q3'][mask_q3], marker='.', linestyle='--', label='Model (q3)')
    if jpeg_mean is not None and not jpeg_mean.empty:
        jm = jpeg_mean[jpeg_mean['bpp_mean']>0]
        if not jm.empty:
            ax.plot(np.log2(jm['bpp_mean'].values), jm['psnr_mean'].values, marker='x', label='JPEG')
    if sm is not None:
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04); cbar.set_label('Normalized reward', rotation=270, labelpad=12)
    ax.set_xlabel('log2(bpp)'); ax.set_ylabel('PSNR (dB)'); ax.set_title(f'RD curve [{kind}] with labels (log2 bpp)')
    ax.grid(True, linestyle='--', alpha=0.4); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out_prefix}_log2_labels.png"); plt.close(fig)

def _adjust_text_no_overlap(ax, texts, iterations=200, step_pixels=1.0, expand=1.1):
    fig = ax.figure; fig.canvas.draw()
    for _ in range(iterations):
        moved = False
        renderer = fig.canvas.get_renderer()
        boxes = [t.get_window_extent(renderer=renderer).expanded(expand, expand) for t in texts]
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                if boxes[i].overlaps(boxes[j]):
                    ci = ((boxes[i].x0 + boxes[i].x1) * 0.5, (boxes[i].y0 + boxes[i].y1) * 0.5)
                    cj = ((boxes[j].x0 + boxes[j].x1) * 0.5, (boxes[j].y0 + boxes[j].y1) * 0.5)
                    dx = ci[0] - cj[0]; dy = ci[1] - cj[1]
                    if dx == 0 and dy == 0: dx = 1.0; dy = 1.0
                    mag = math.hypot(dx, dy); ux = dx / mag; uy = dy / mag
                    pi = texts[i].get_position(); pj = texts[j].get_position()
                    pi_disp = ax.transData.transform(pi); pj_disp = ax.transData.transform(pj)
                    pi_disp = (pi_disp[0] + ux * step_pixels, pi_disp[1] + uy * step_pixels)
                    pj_disp = (pj_disp[0] - ux * step_pixels, pj_disp[1] - uy * step_pixels)
                    texts[i].set_position(ax.transData.inverted().transform(pi_disp))
                    texts[j].set_position(ax.transData.inverted().transform(pj_disp))
                    moved = True
        if not moved: break
    fig.canvas.draw()

def plot_rd_with_textlabels(df_model, kind, out_prefix, jpeg_mean=None):
    d = df_model[df_model['kind']==kind].sort_values(['epoch','batch'])
    if d.empty: return
    labels = [f"{e}-{b}" for e,b in zip(d['epoch'].values, d['batch'].values)]
    cols, norm, cmap = _norm_colors(d['reward'].values)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap) if norm is not None else None
    if sm is not None: sm.set_array([])
    fig = plt.figure(figsize=(7,5), dpi=140); ax = plt.gca()
    texts = []
    for xi, yi, lab, c in zip(d['bpp_med'].values, d['psnr_med'].values, labels, cols):
        texts.append(ax.text(xi, yi, lab, fontsize=6, ha='center', va='center', color=c))
    if jpeg_mean is not None and not jpeg_mean.empty:
        ax.plot(jpeg_mean['bpp_mean'].values, jpeg_mean['psnr_mean'].values, marker='x', label='JPEG')
    ax.plot(d['bpp_q1'], d['psnr_q1'], marker='.', linestyle='--', label='Model (q1)')
    ax.plot(d['bpp_q3'], d['psnr_q3'], marker='.', linestyle='--', label='Model (q3)')
    _adjust_text_no_overlap(ax, texts, iterations=300, step_pixels=0.8, expand=1.1)
    if sm is not None:
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04); cbar.set_label('Normalized reward', rotation=270, labelpad=12)
    ax.set_xlabel('bpp'); ax.set_ylabel('PSNR (dB)'); ax.set_title(f'RD curve [{kind}] text labels (linear)')
    ax.grid(True, linestyle='--', alpha=0.4)
    if jpeg_mean is not None and not jpeg_mean.empty: ax.legend()
    fig.tight_layout(); fig.savefig(f"{out_prefix}_linear_textlabels.png"); plt.close(fig)

    mask_med = d['bpp_med'].values > 0
    fig = plt.figure(figsize=(7,5), dpi=140); ax = plt.gca(); texts = []
    if mask_med.any():
        lx = np.log2(d['bpp_med'][mask_med]); ly = d['psnr_med'][mask_med]
        llabels = list(np.array(labels)[mask_med]); lcols = list(np.array(cols)[mask_med])
        for xi, yi, lab, c in zip(lx, ly, llabels, lcols):
            texts.append(ax.text(xi, yi, lab, fontsize=6, ha='center', va='center', color=c))
    if jpeg_mean is not None and not jpeg_mean.empty:
        jm = jpeg_mean[jpeg_mean['bpp_mean']>0]
        if not jm.empty:
            ax.plot(np.log2(jm['bpp_mean'].values), jm['psnr_mean'].values, marker='x', label='JPEG')
    mask_q1 = d['bpp_q1'].values > 0; mask_q3 = d['bpp_q3'].values > 0
    if mask_q1.any():
        ax.plot(np.log2(d['bpp_q1'][mask_q1]), d['psnr_q1'][mask_q1], marker='.', linestyle='--', label='Model (q1)')
    if mask_q3.any():
        ax.plot(np.log2(d['bpp_q3'][mask_q3]), d['psnr_q3'][mask_q3], marker='.', linestyle='--', label='Model (q3)')
    _adjust_text_no_overlap(ax, texts, iterations=300, step_pixels=0.8, expand=1.1)
    if sm is not None:
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04); cbar.set_label('Normalized reward', rotation=270, labelpad=12)
    ax.set_xlabel('log2(bpp)'); ax.set_ylabel('PSNR (dB)'); ax.set_title(f'RD curve [{kind}] text labels (log2 bpp)')
    ax.grid(True, linestyle='--', alpha=0.4)
    if jpeg_mean is not None and not jpeg_mean.empty: ax.legend()
    fig.tight_layout(); fig.savefig(f"{out_prefix}_log2_textlabels.png"); plt.close(fig)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default='logs/model.out', help='primary log path or directory')
    ap.add_argument('--also', default='logs/model.err', help='secondary log path')
    ap.add_argument('--outdir', default='plots', help='output directory')
    ap.add_argument('--imgs', default='../../../dataset/ambi/clic2024_split/train_100/', help='image dir for JPEG RD')
    ap.add_argument('--qualities', default='5,10,20,30,40,50,60,70,80,90,95', help='JPEG qualities csv')
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    p1 = Path(args.log)
    p2 = Path(args.also)
    paths = []
    if p1.is_dir():
        for pat in ('*.out','*.err','*.log','*.out.gz','*.err.gz','*.log.gz'):
            paths.extend(sorted(p1.glob(pat)))
    else:
        if p1.exists(): paths.append(p1)
        if p2.exists(): paths.append(p2)

    df = parse_log_multi(paths)
    if df.empty:
        print("no records parsed"); return

    parsed_csv = outdir / 'parsed_rd.csv'
    df.to_csv(parsed_csv, index=False)

    img_dir = args.imgs
    qualities = [int(x) for x in args.qualities.split(',') if x.strip()]
    per_image_csv = outdir / "jpeg_rd_per_image.csv"
    mean_csv = outdir / "jpeg_rd_mean.csv"
    if per_image_csv.exists() and mean_csv.exists():
        per_image = pd.read_csv(per_image_csv)
        mean_df = pd.read_csv(mean_csv)
    else:
        per_image = jpeg_rd_per_image(img_dir, qualities)
        mean_df = jpeg_rd_mean(per_image)
        per_image.to_csv(per_image_csv, index=False)
        mean_df.to_csv(mean_csv, index=False)

    plot_rd_with_jpeg(df, 'TOTAL', str(outdir/'rd_total'), jpeg_mean=mean_df)
    plot_rd_with_jpeg(df, 'BATCH', str(outdir/'rd_batch'), jpeg_mean=mean_df)
    plot_rd_with_labels(df, 'TOTAL', str(outdir/'rd_total'), jpeg_mean=mean_df)
    plot_rd_with_labels(df, 'BATCH', str(outdir/'rd_batch'), jpeg_mean=mean_df)
    plot_rd_with_textlabels(df, 'TOTAL', str(outdir/'rd_total'), jpeg_mean=mean_df)
    plot_rd_with_textlabels(df, 'BATCH', str(outdir/'rd_batch'), jpeg_mean=mean_df)

    print(f"Saved: {parsed_csv}")
    print(f"Saved: {per_image_csv}")
    print(f"Saved: {mean_csv}")
    print(f"Plots written to {outdir}")

if __name__ == "__main__":
    main()
