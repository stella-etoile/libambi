#!/usr/bin/env python3
import os, math, tempfile, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def psnr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    mse = ((a - b) ** 2).mean()
    if mse <= 0:
        return 120.0
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)

def load_per_batch(csv_path):
    df = pd.read_csv(csv_path)
    df["epoch"] = df["epoch"].astype(int)
    df["bpp_med"] = df["bpp_med"].astype(float)
    df["psnr_med"] = df["psnr_med"].astype(float)
    df["R"] = df["R"].astype(float)
    df["phase"] = df["phase"].astype(str)
    df["batch"] = df["batch"].astype(str)
    return df

def ensure_outdir(d):
    d = Path(d); d.mkdir(parents=True, exist_ok=True); return d

def transform_bpp(x, mode):
    x = np.asarray(x, dtype=np.float64)
    if mode == "linear": return x
    if mode == "log2":   return np.log2(np.maximum(x, 1e-9))
    return x

def norm_colors(v, vmin, vmax):
    v = np.asarray(v, dtype=np.float64)
    if vmax == vmin: return np.full_like(v, 0.5)
    return (v - vmin) / (vmax - vmin)

def compute_limits(x_arrays, y_arrays, pad_ratio=0.05):
    xs = np.concatenate([np.asarray(a).ravel() for a in x_arrays if a is not None and len(a) > 0])
    ys = np.concatenate([np.asarray(a).ravel() for a in y_arrays if a is not None and len(a) > 0])
    xs = xs[np.isfinite(xs)]; ys = ys[np.isfinite(ys)]
    if xs.size == 0 or ys.size == 0: return None, None
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    if xmax == xmin: xmax += 1e-6
    if ymax == ymin: ymax += 1e-6
    xpad = (xmax - xmin) * pad_ratio; ypad = (ymax - ymin) * pad_ratio
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)

def jpeg_quantiles_curve(img_dir, qualities, sample, seed=1337):
    rng = np.random.default_rng(seed)
    img_dir = Path(img_dir)
    files = sorted([p for p in img_dir.glob("*") if p.is_file()])
    if sample is not None and sample > 0 and sample < len(files):
        idx = rng.choice(len(files), size=sample, replace=False)
        files = [files[i] for i in idx]
    tmp = Path(tempfile.mkdtemp(prefix="jpeg_tmp_"))
    rows = []
    for q in qualities:
        bpps, psnrs = [], []
        for f in files:
            try:
                im = Image.open(f).convert("RGB")
                w, h = im.size
                outp = tmp / f"{f.stem}_q{q}.jpg"
                im.save(outp, format="JPEG", quality=int(q), subsampling=0, optimize=True)
                bpp = (outp.stat().st_size * 8.0) / (w * h)
                im_j = Image.open(outp).convert("RGB")
                p = psnr(im, im_j)
                bpps.append(bpp); psnrs.append(p)
            except Exception:
                pass
        if bpps:
            bpps = np.array(bpps, float); psnrs = np.array(psnrs, float)
            rows.append({
                "quality": q,
                "bpp_min": float(np.min(bpps)),
                "bpp_q1": float(np.quantile(bpps, 0.25)),
                "bpp_med": float(np.quantile(bpps, 0.50)),
                "bpp_q3": float(np.quantile(bpps, 0.75)),
                "bpp_max": float(np.max(bpps)),
                "psnr_min": float(np.min(psnrs)),
                "psnr_q1": float(np.quantile(psnrs, 0.25)),
                "psnr_med": float(np.quantile(psnrs, 0.50)),
                "psnr_q3": float(np.quantile(psnrs, 0.75)),
                "psnr_max": float(np.max(psnrs)),
            })
    shutil.rmtree(tmp, ignore_errors=True)
    return pd.DataFrame(rows).sort_values("quality")

def truncated_cmap(name="nipy_spectral", vmin=0.02, vmax=0.85, n=256):
    base = plt.get_cmap(name)
    vals = base(np.linspace(vmin, vmax, n))
    return matplotlib.colors.ListedColormap(vals)

CMAP = truncated_cmap()

def overlay_jpeg_q_lines(ax, jpeg_df, xmode):
    if jpeg_df is None or len(jpeg_df) == 0:
        return
    x_min = transform_bpp(jpeg_df["bpp_min"].values, xmode)
    y_min = jpeg_df["psnr_min"].values
    x_med = transform_bpp(jpeg_df["bpp_med"].values, xmode)
    y_med = jpeg_df["psnr_med"].values
    x_max = transform_bpp(jpeg_df["bpp_max"].values, xmode)
    y_max = jpeg_df["psnr_max"].values
    ax.plot(x_min, y_min, linewidth=2.0, color="black", linestyle="--", label="JPEG min")
    ax.plot(x_med, y_med, linewidth=2.5, color="green", label="JPEG median")
    ax.plot(x_max, y_max, linewidth=2.0, color="black", linestyle="-", label="JPEG max")
    ax.legend(loc="best", fontsize=8)

def draw_labels(ax, x, y, labels, cvals, title, xlabel, ylabel):
    norm = plt.Normalize(vmin=0, vmax=1)
    for xi, yi, li, cv in zip(x, y, labels, cvals):
        ax.text(xi, yi, str(li), color=CMAP(norm(cv)), fontsize=8, ha="center", va="center")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    cb = plt.colorbar(sm, ax=ax); cb.set_label("Reward (low→high)")

def plot_epoch_batches(df_b, epoch, phases, xmode, outdir, jpeg_df):
    dfe = df_b[(df_b["epoch"] == epoch) & (df_b["phase"].isin(phases))]
    if len(dfe) == 0: return
    v = dfe["R"].values; cvals = norm_colors(v, float(np.min(v)), float(np.max(v)))
    x = transform_bpp(dfe["bpp_med"].values, xmode); y = dfe["psnr_med"].values
    labels = dfe["batch"].values
    fig, ax = plt.subplots(figsize=(7,5))
    draw_labels(ax, x, y, labels, cvals, f"Epoch {epoch} ({','.join(phases)}) [{xmode}]",
                "bpp_log2" if xmode=="log2" else "bpp", "PSNR")
    overlay_jpeg_q_lines(ax, jpeg_df, xmode)
    xlim, ylim = compute_limits(
        [
            x,
            transform_bpp(jpeg_df["bpp_min"].values, xmode) if jpeg_df is not None and len(jpeg_df)>0 else None,
            transform_bpp(jpeg_df["bpp_med"].values, xmode) if jpeg_df is not None and len(jpeg_df)>0 else None,
            transform_bpp(jpeg_df["bpp_max"].values, xmode) if jpeg_df is not None and len(jpeg_df)>0 else None,
        ],
        [
            y,
            jpeg_df["psnr_min"].values if jpeg_df is not None and len(jpeg_df)>0 else None,
            jpeg_df["psnr_med"].values if jpeg_df is not None and len(jpeg_df)>0 else None,
            jpeg_df["psnr_max"].values if jpeg_df is not None and len(jpeg_df)>0 else None,
        ],
        pad_ratio=0.06
    )
    if xlim and ylim: ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    fig.tight_layout(); fig.savefig(outdir / f"epoch_{epoch}_{'-'.join(phases)}_{xmode}.png", dpi=150); plt.close(fig)

def plot_all_epochs(df_b, phases, xmode, outdir, jpeg_df):
    dfa = df_b[df_b["phase"].isin(phases)]
    if len(dfa) == 0: return
    v = dfa["R"].values; cvals = norm_colors(v, float(np.min(v)), float(np.max(v)))
    x = transform_bpp(dfa["bpp_med"].values, xmode); y = dfa["psnr_med"].values
    labels = dfa["epoch"].values
    fig, ax = plt.subplots(figsize=(7,5))
    draw_labels(ax, x, y, labels, cvals, f"All epochs ({','.join(phases)}) [{xmode}]",
                "bpp_log2" if xmode=="log2" else "bpp", "PSNR")
    overlay_jpeg_q_lines(ax, jpeg_df, xmode)
    xlim, ylim = compute_limits(
        [
            x,
            transform_bpp(jpeg_df["bpp_min"].values, xmode) if jpeg_df is not None and len(jpeg_df)>0 else None,
            transform_bpp(jpeg_df["bpp_med"].values, xmode) if jpeg_df is not None and len(jpeg_df)>0 else None,
            transform_bpp(jpeg_df["bpp_max"].values, xmode) if jpeg_df is not None and len(jpeg_df)>0 else None,
        ],
        [
            y,
            jpeg_df["psnr_min"].values if jpeg_df is not None and len(jpeg_df)>0 else None,
            jpeg_df["psnr_med"].values if jpeg_df is not None and len(jpeg_df)>0 else None,
            jpeg_df["psnr_max"].values if jpeg_df is not None and len(jpeg_df)>0 else None,
        ],
        pad_ratio=0.06
    )
    if xlim and ylim: ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    fig.tight_layout(); fig.savefig(outdir / f"all_epochs_{'-'.join(phases)}_{xmode}.png", dpi=150); plt.close(fig)

def plot_all_batches(df_b, xmode, outdir, jpeg_df):
    v = df_b["R"].values; cvals = norm_colors(v, float(np.min(v)), float(np.max(v)))
    x = transform_bpp(df_b["bpp_med"].values, xmode); y = df_b["psnr_med"].values
    labels = [f"{e}:{b}" for e, b in zip(df_b["epoch"].values, df_b["batch"].values)]
    fig, ax = plt.subplots(figsize=(7,5))
    draw_labels(ax, x, y, labels, cvals, f"All batches (train+val) [{xmode}]",
                "bpp_log2" if xmode=="log2" else "bpp", "PSNR")
    overlay_jpeg_q_lines(ax, jpeg_df, xmode)
    xlim, ylim = compute_limits(
        [
            x,
            transform_bpp(jpeg_df["bpp_min"].values, xmode) if jpeg_df is not None and len(jpeg_df)>0 else None,
            transform_bpp(jpeg_df["bpp_med"].values, xmode) if jpeg_df is not None and len(jpeg_df)>0 else None,
            transform_bpp(jpeg_df["bpp_max"].values, xmode) if jpeg_df is not None and len(jpeg_df)>0 else None,
        ],
        [
            y,
            jpeg_df["psnr_min"].values if jpeg_df is not None and len(jpeg_df)>0 else None,
            jpeg_df["psnr_med"].values if jpeg_df is not None and len(jpeg_df)>0 else None,
            jpeg_df["psnr_max"].values if jpeg_df is not None and len(jpeg_df)>0 else None,
        ],
        pad_ratio=0.06
    )
    if xlim and ylim: ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    fig.tight_layout(); fig.savefig(outdir / f"all_batches_{xmode}.png", dpi=150); plt.close(fig)

def main():
    per_batch = "logs/per_batch.csv"
    outdir = ensure_outdir("plots_bpp_psnr")
    jpeg_dir = "/mnt/Jupiter/dataset/ambi/clic2024_split/train_200"
    jpeg_qualities = [0,5,10,20,30,40,50,60,70,80,90,95,100]
    jpeg_sample = 64

    df_b = load_per_batch(per_batch)
    jpeg_df = jpeg_quantiles_curve(jpeg_dir, jpeg_qualities, jpeg_sample)

    epochs = sorted(df_b["epoch"].unique().tolist())
    for e in epochs:
        for ph in [["train"], ["val"], ["train","val"]]:
            for xm in ["linear","log2"]:
                plot_epoch_batches(df_b, e, ph, xm, outdir, jpeg_df)

    for ph in [["train"], ["val"], ["train","val"]]:
        for xm in ["linear","log2"]:
            plot_all_epochs(df_b, ph, xm, outdir, jpeg_df)

    for xm in ["linear","log2"]:
        plot_all_batches(df_b, xm, outdir, jpeg_df)

if __name__ == "__main__":
    main()