# parse_and_plot_rd.py
import re, math, io
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

PAT1 = re.compile(
    r'^\[Epoch\s+(?P<epoch>\d+)\/(?P<epochs>\d+)\]\s+'
    r'(?P<batch>\d+)\/(?P<nbatch>~?\d+)\s+'
    r'(?P<kind>BATCH|TOTAL)\s+'
    r'bpp=\[(?P<bpp>[^\]]+)\]\s+'
    r'PSNR=\[(?P<psnr>[^\]]+)\]\s+'
    r'SSIM=(?P<ssim>[-+eE\d\.]+)\s+'
    r'MS=(?P<ms>[-+eE\d\.]+)\s+'
    r'R~?=(?P<r>[-+eE\d\.]+)\s*$'
)

PAT2 = re.compile(
    r'^\s*(?P<epoch>\d+)\s*\|\s*(?P<batch>\d+)\/(?P<nbatch>\d+)\s*\|\s*\[(?P<bpp>[^\]]+)\]\s*\|\s*\[(?P<psnr>[^\]]+)\]\s*\|\s*(?P<ssim>[-+eE\d\.]+)\s*\|\s*(?P<ms>[-+eE\d\.]+)\s*\|\s*(?P<r>[-+eE\d\.]+)\s*$'
)

def parse_five(x):
    xs = [float(t.strip()) for t in x.split(',')]
    return xs if len(xs) == 5 else [np.nan]*5

def make_row(m, kind1):
    if kind1:
        epoch = int(m.group('epoch'))
        kind = m.group('kind')
    else:
        epoch = int(m.group('epoch'))
        kind = "BATCH"
    batch = int(m.group('batch'))
    bpp = parse_five(m.group('bpp'))
    psnr = parse_five(m.group('psnr'))
    r = float(m.group('r'))
    return dict(
        epoch=epoch, batch=batch, kind=kind,
        bpp_min=bpp[0], bpp_q1=bpp[1], bpp_med=bpp[2], bpp_q3=bpp[3], bpp_max=bpp[4],
        psnr_min=psnr[0], psnr_q1=psnr[1], psnr_med=psnr[2], psnr_q3=psnr[3], psnr_max=psnr[4],
        reward=r
    )

def parse_log(p):
    rows = []
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m1 = PAT1.match(line)
            if m1:
                rows.append(make_row(m1, True))
                continue
            m2 = PAT2.match(line)
            if m2:
                rows.append(make_row(m2, False))
    return pd.DataFrame(rows)

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
    files = sorted(files)
    return files

def jpeg_rd_per_image(img_dir, qualities):
    files = list_images(img_dir)
    rows = []
    for q in qualities:
        for f in files:
            img = Image.open(f).convert('RGB')
            w,h = img.size
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=q)
            size = len(buf.getvalue())
            bpp = (size*8)/(w*h)
            buf.seek(0)
            rec = Image.open(buf)
            rows.append(dict(file=str(f.name), quality=q, bpp=bpp, psnr=psnr(img, rec)))
    return pd.DataFrame(rows)

def jpeg_rd_stats(per_image_df):
    q1 = 0.25
    q3 = 0.75
    g = per_image_df.groupby('quality').agg(
        bpp_mean=('bpp','mean'),
        bpp_std=('bpp','std'),
        bpp_q1=('bpp', lambda s: s.quantile(q1)),
        bpp_med=('bpp', lambda s: s.quantile(0.5)),
        bpp_q3=('bpp', lambda s: s.quantile(q3)),
        psnr_mean=('psnr','mean'),
        psnr_std=('psnr','std'),
        psnr_q1=('psnr', lambda s: s.quantile(q1)),
        psnr_med=('psnr', lambda s: s.quantile(0.5)),
        psnr_q3=('psnr', lambda s: s.quantile(q3)),
        n=('psnr','count')
    ).reset_index()
    return g.sort_values('quality')

def plot_rd_with_jpeg(df_model, kind, out_prefix, jpeg_stats=None):
    d = df_model[df_model['kind']==kind].sort_values(['epoch','batch'])
    if d.empty:
        return
    fig = plt.figure(figsize=(7,5), dpi=140)
    plt.plot(d['bpp_med'], d['psnr_med'], marker='o', label='Model (med)')
    plt.plot(d['bpp_q1'], d['psnr_q1'], marker='.', linestyle='--', label='Model (q1)')
    plt.plot(d['bpp_q3'], d['psnr_q3'], marker='.', linestyle='--', label='Model (q3)')
    if jpeg_stats is not None and not jpeg_stats.empty:
        plt.plot(jpeg_stats['bpp_med'], jpeg_stats['psnr_med'], marker='x', label='JPEG (med)')
        plt.plot(jpeg_stats['bpp_q1'], jpeg_stats['psnr_q1'], marker='x', linestyle='--', label='JPEG (q1)')
        plt.plot(jpeg_stats['bpp_q3'], jpeg_stats['psnr_q3'], marker='x', linestyle='--', label='JPEG (q3)')
    plt.xlabel('bpp')
    plt.ylabel('PSNR (dB)')
    plt.title(f'RD curve [{kind}] (linear)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_linear_with_jpeg.png")
    plt.close(fig)
    mask_med = d['bpp_med'].values > 0
    mask_q1 = d['bpp_q1'].values > 0
    mask_q3 = d['bpp_q3'].values > 0
    fig = plt.figure(figsize=(7,5), dpi=140)
    if mask_med.any():
        plt.plot(np.log2(d['bpp_med'][mask_med]), d['psnr_med'][mask_med], marker='o', label='Model (med)')
    if mask_q1.any():
        plt.plot(np.log2(d['bpp_q1'][mask_q1]), d['psnr_q1'][mask_q1], marker='.', linestyle='--', label='Model (q1)')
    if mask_q3.any():
        plt.plot(np.log2(d['bpp_q3'][mask_q3]), d['psnr_q3'][mask_q3], marker='.', linestyle='--', label='Model (q3)')
    if jpeg_stats is not None and not jpeg_stats.empty:
        jm = jpeg_stats.copy()
        jm = jm[jm['bpp_med']>0]
        if not jm.empty:
            plt.plot(np.log2(jm['bpp_med']), jm['psnr_med'], marker='x', label='JPEG (med)')
        jq1 = jpeg_stats.copy()
        jq1 = jq1[jq1['bpp_q1']>0]
        if not jq1.empty:
            plt.plot(np.log2(jq1['bpp_q1']), jq1['psnr_q1'], marker='x', linestyle='--', label='JPEG (q1)')
        jq3 = jpeg_stats.copy()
        jq3 = jq3[jq3['bpp_q3']>0]
        if not jq3.empty:
            plt.plot(np.log2(jq3['bpp_q3']), jq3['psnr_q3'], marker='x', linestyle='--', label='JPEG (q3)')
    plt.xlabel('log2(bpp)')
    plt.ylabel('PSNR (dB)')
    plt.title(f'RD curve [{kind}] (log2 bpp)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_log2_with_jpeg.png")
    plt.close(fig)

def plot_rd_with_labels(df_model, kind, out_prefix, jpeg_stats=None):
    d = df_model[df_model['kind']==kind].sort_values(['epoch','batch'])
    if d.empty:
        return
    labels = [f"{e}-{b}" for e,b in zip(d['epoch'].values, d['batch'].values)]
    fig = plt.figure(figsize=(7,5), dpi=140)
    plt.plot(d['bpp_med'], d['psnr_med'], marker='o', label='Model (med)')
    for xi, yi, lab in zip(d['bpp_med'].values, d['psnr_med'].values, labels):
        plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(5,5), fontsize=7)
    plt.plot(d['bpp_q1'], d['psnr_q1'], marker='.', linestyle='--', label='Model (q1)')
    plt.plot(d['bpp_q3'], d['psnr_q3'], marker='.', linestyle='--', label='Model (q3)')
    if jpeg_stats is not None and not jpeg_stats.empty:
        plt.plot(jpeg_stats['bpp_med'], jpeg_stats['psnr_med'], marker='x', label='JPEG (med)')
        plt.plot(jpeg_stats['bpp_q1'], jpeg_stats['psnr_q1'], marker='x', linestyle='--', label='JPEG (q1)')
        plt.plot(jpeg_stats['bpp_q3'], jpeg_stats['psnr_q3'], marker='x', linestyle='--', label='JPEG (q3)')
    plt.xlabel('bpp')
    plt.ylabel('PSNR (dB)')
    plt.title(f'RD curve [{kind}] with labels (linear)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_linear_labels.png")
    plt.close(fig)
    mask_med = d['bpp_med'].values > 0
    fig = plt.figure(figsize=(7,5), dpi=140)
    if mask_med.any():
        lx = np.log2(d['bpp_med'][mask_med])
        ly = d['psnr_med'][mask_med]
        llabels = list(np.array(labels)[mask_med])
        plt.plot(lx, ly, marker='o', label='Model (med)')
        for xi, yi, lab in zip(lx, ly, llabels):
            plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(5,5), fontsize=7)
    mask_q1 = d['bpp_q1'].values > 0
    mask_q3 = d['bpp_q3'].values > 0
    if mask_q1.any():
        plt.plot(np.log2(d['bpp_q1'][mask_q1]), d['psnr_q1'][mask_q1], marker='.', linestyle='--', label='Model (q1)')
    if mask_q3.any():
        plt.plot(np.log2(d['bpp_q3'][mask_q3]), d['psnr_q3'][mask_q3], marker='.', linestyle='--', label='Model (q3)')
    if jpeg_stats is not None and not jpeg_stats.empty:
        jm = jpeg_stats[jpeg_stats['bpp_med']>0]
        if not jm.empty:
            plt.plot(np.log2(jm['bpp_med']), jm['psnr_med'], marker='x', label='JPEG (med)')
        jq1 = jpeg_stats[jpeg_stats['bpp_q1']>0]
        if not jq1.empty:
            plt.plot(np.log2(jq1['bpp_q1']), jq1['psnr_q1'], marker='x', linestyle='--', label='JPEG (q1)')
        jq3 = jpeg_stats[jpeg_stats['bpp_q3']>0]
        if not jq3.empty:
            plt.plot(np.log2(jq3['bpp_q3']), jq3['psnr_q3'], marker='x', linestyle='--', label='JPEG (q3)')
    plt.xlabel('log2(bpp)')
    plt.ylabel('PSNR (dB)')
    plt.title(f'RD curve [{kind}] with labels (log2 bpp)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_log2_labels.png")
    plt.close(fig)

def _adjust_text_no_overlap(ax, texts, iterations=200, step_pixels=1.0, expand=1.1):
    fig = ax.figure
    fig.canvas.draw()
    for _ in range(iterations):
        moved = False
        renderer = fig.canvas.get_renderer()
        boxes = [t.get_window_extent(renderer=renderer).expanded(expand, expand) for t in texts]
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                if boxes[i].overlaps(boxes[j]):
                    ci = ((boxes[i].x0 + boxes[i].x1) * 0.5, (boxes[i].y0 + boxes[i].y1) * 0.5)
                    cj = ((boxes[j].x0 + boxes[j].x1) * 0.5, (boxes[j].y0 + boxes[j].y1) * 0.5)
                    dx = ci[0] - cj[0]
                    dy = ci[1] - cj[1]
                    if dx == 0 and dy == 0:
                        dx = 1.0
                        dy = 1.0
                    mag = math.hypot(dx, dy)
                    ux = dx / mag
                    uy = dy / mag
                    pi = texts[i].get_position()
                    pj = texts[j].get_position()
                    pi_disp = ax.transData.transform(pi)
                    pj_disp = ax.transData.transform(pj)
                    pi_disp = (pi_disp[0] + ux * step_pixels, pi_disp[1] + uy * step_pixels)
                    pj_disp = (pj_disp[0] - ux * step_pixels, pj_disp[1] - uy * step_pixels)
                    texts[i].set_position(ax.transData.inverted().transform(pi_disp))
                    texts[j].set_position(ax.transData.inverted().transform(pj_disp))
                    moved = True
        if not moved:
            break
    fig.canvas.draw()

def plot_rd_with_textlabels(df_model, kind, out_prefix, jpeg_stats=None):
    d = df_model[df_model['kind']==kind].sort_values(['epoch','batch'])
    if d.empty:
        return
    labels = [f"{e}-{b}" for e,b in zip(d['epoch'].values, d['batch'].values)]
    rewards = d['reward'].values
    if np.all(np.isnan(rewards)):
        rewards = np.zeros_like(rewards)
    norm = mcolors.Normalize(vmin=np.nanmin(rewards), vmax=np.nanmax(rewards))
    cmap = cm.get_cmap('RdYlGn')
    fig = plt.figure(figsize=(7,5), dpi=140)
    ax = plt.gca()
    texts = []
    for xi, yi, lab, rv in zip(d['bpp_med'].values, d['psnr_med'].values, labels, rewards):
        color = cmap(norm(rv))
        texts.append(ax.text(xi, yi, lab, fontsize=7, ha='center', va='center', color=color))
    if jpeg_stats is not None and not jpeg_stats.empty:
        plt.plot(jpeg_stats['bpp_med'].values, jpeg_stats['psnr_med'].values, marker='x', label='JPEG (med)')
        plt.plot(jpeg_stats['bpp_q1'].values, jpeg_stats['psnr_q1'].values, marker='x', linestyle='--', label='JPEG (q1)')
        plt.plot(jpeg_stats['bpp_q3'].values, jpeg_stats['psnr_q3'].values, marker='x', linestyle='--', label='JPEG (q3)')
    plt.plot(d['bpp_q1'], d['psnr_q1'], marker='.', linestyle='--', label='Model (q1)')
    plt.plot(d['bpp_q3'], d['psnr_q3'], marker='.', linestyle='--', label='Model (q3)')
    _adjust_text_no_overlap(ax, texts, iterations=300, step_pixels=0.8, expand=1.1)
    plt.xlabel('bpp')
    plt.ylabel('PSNR (dB)')
    plt.title(f'RD curve [{kind}] text labels colored by reward (linear)')
    plt.grid(True, linestyle='--', alpha=0.4)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Reward')
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_linear_textlabels.png")
    plt.close(fig)
    mask_med = d['bpp_med'].values > 0
    fig = plt.figure(figsize=(7,5), dpi=140)
    ax = plt.gca()
    texts = []
    if mask_med.any():
        lx = np.log2(d['bpp_med'][mask_med])
        ly = d['psnr_med'][mask_med]
        llabels = list(np.array(labels)[mask_med])
        lrewards = rewards[mask_med]
        for xi, yi, lab, rv in zip(lx, ly, llabels, lrewards):
            color = cmap(norm(rv))
            texts.append(ax.text(xi, yi, lab, fontsize=7, ha='center', va='center', color=color))
    if jpeg_stats is not None and not jpeg_stats.empty:
        jm = jpeg_stats[jpeg_stats['bpp_med']>0]
        if not jm.empty:
            plt.plot(np.log2(jm['bpp_med'].values), jm['psnr_med'].values, marker='x', label='JPEG (med)')
        jq1 = jpeg_stats[jpeg_stats['bpp_q1']>0]
        if not jq1.empty:
            plt.plot(np.log2(jq1['bpp_q1'].values), jq1['psnr_q1'].values, marker='x', linestyle='--', label='JPEG (q1)')
        jq3 = jpeg_stats[jpeg_stats['bpp_q3']>0]
        if not jq3.empty:
            plt.plot(np.log2(jq3['bpp_q3'].values), jq3['psnr_q3'].values, marker='x', linestyle='--', label='JPEG (q3)')
    mask_q1 = d['bpp_q1'].values > 0
    mask_q3 = d['bpp_q3'].values > 0
    if mask_q1.any():
        plt.plot(np.log2(d['bpp_q1'][mask_q1]), d['psnr_q1'][mask_q1], marker='.', linestyle='--', label='Model (q1)')
    if mask_q3.any():
        plt.plot(np.log2(d['bpp_q3'][mask_q3]), d['psnr_q3'][mask_q3], marker='.', linestyle='--', label='Model (q3)')
    _adjust_text_no_overlap(ax, texts, iterations=300, step_pixels=0.8, expand=1.1)
    plt.xlabel('log2(bpp)')
    plt.ylabel('PSNR (dB)')
    plt.title(f'RD curve [{kind}] text labels colored by reward (log2 bpp)')
    plt.grid(True, linestyle='--', alpha=0.4)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Reward')
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_log2_textlabels.png")
    plt.close(fig)

def _ensure_jpeg_stats(img_dir, qualities, per_image_csv, mean_csv):
    per_image = None
    mean_df = None
    if per_image_csv.exists():
        per_image = pd.read_csv(per_image_csv)
    if mean_csv.exists():
        mean_df = pd.read_csv(mean_csv)
    needed = {'bpp_q1','bpp_med','bpp_q3','psnr_q1','psnr_med','psnr_q3'}
    if mean_df is not None and needed.issubset(set(mean_df.columns)):
        return per_image, mean_df
    if per_image is None:
        per_image = jpeg_rd_per_image(img_dir, qualities)
        per_image.to_csv(per_image_csv, index=False)
    mean_df = jpeg_rd_stats(per_image)
    mean_df.to_csv(mean_csv, index=False)
    return per_image, mean_df

def main():
    log_path = Path("logs/model.out")
    outdir = Path("plots")
    outdir.mkdir(parents=True, exist_ok=True)
    df = parse_log(log_path)
    if df.empty:
        print("no records parsed"); return
    img_dir = "../../../dataset/ambi/clic2024_split/train_100/"
    qualities = [5,10,20,30,40,50,60,70,80,90,95]
    per_image_csv = outdir / "jpeg_rd_per_image.csv"
    mean_csv = outdir / "jpeg_rd_mean.csv"
    per_image, mean_df = _ensure_jpeg_stats(img_dir, qualities, per_image_csv, mean_csv)
    plot_rd_with_jpeg(df, 'TOTAL', str(outdir/'rd_total'), jpeg_stats=mean_df)
    plot_rd_with_jpeg(df, 'BATCH', str(outdir/'rd_batch'), jpeg_stats=mean_df)
    plot_rd_with_labels(df, 'TOTAL', str(outdir/'rd_total'), jpeg_stats=mean_df)
    plot_rd_with_labels(df, 'BATCH', str(outdir/'rd_batch'), jpeg_stats=mean_df)
    plot_rd_with_textlabels(df, 'TOTAL', str(outdir/'rd_total'), jpeg_stats=mean_df)
    plot_rd_with_textlabels(df, 'BATCH', str(outdir/'rd_batch'), jpeg_stats=mean_df)
    epochs = sorted(df['epoch'].unique().tolist())
    for e in epochs:
        edf = df[df['epoch']==e].copy()
        eout = outdir / f"epoch_{e:03d}"
        eout.mkdir(parents=True, exist_ok=True)
        plot_rd_with_jpeg(edf, 'TOTAL', str(eout/'rd_total'), jpeg_stats=mean_df)
        plot_rd_with_jpeg(edf, 'BATCH', str(eout/'rd_batch'), jpeg_stats=mean_df)
        plot_rd_with_labels(edf, 'TOTAL', str(eout/'rd_total'), jpeg_stats=mean_df)
        plot_rd_with_labels(edf, 'BATCH', str(eout/'rd_batch'), jpeg_stats=mean_df)
        plot_rd_with_textlabels(edf, 'TOTAL', str(eout/'rd_total'), jpeg_stats=mean_df)
        plot_rd_with_textlabels(edf, 'BATCH', str(eout/'rd_batch'), jpeg_stats=mean_df)
    print(f"Saved: {per_image_csv}")
    print(f"Saved: {mean_csv}")
    print(f"Plots written to {outdir}")

if __name__ == "__main__":
    main()
