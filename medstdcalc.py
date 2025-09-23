import sys, os
import pandas as pd
import numpy as np

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    for c in candidates:
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    for col in df.columns:
        if any(k in col.lower() for k in candidates):
            return col
    return None

def load_df():
    paths = [
        "plots/jpeg_rd_per_image.csv",
        "jpeg_rd_per_image.csv",
        "plots/jpeg_rd_mean.csv",
        "jpeg_rd_mean.csv",
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p), p
    raise FileNotFoundError("Could not find any of: " + ", ".join(paths))

def interp_val(x, y, x0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 2: 
        return None
    order = np.argsort(x)
    x = x[order]; y = y[order]
    if x0 < x.min() or x0 > x.max():
        return None
    return float(np.interp(x0, x, y))

def summarize(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return None, None, 0
    return float(np.median(a)), float(np.std(a, ddof=0)), int(a.size)

df, used_path = load_df()

key_candidates = ["image", "img", "file", "filename", "name", "path", "id"]
bpp_candidates = ["bpp", "jpeg_bpp", "bits_per_pixel"]
psnr_candidates = ["psnr", "jpeg_psnr"]

key_col = pick_col(df, key_candidates)
bpp_col = pick_col(df, bpp_candidates)
psnr_col = pick_col(df, psnr_candidates)

if bpp_col is None or psnr_col is None:
    raise KeyError(f"Could not detect bpp/psnr columns. Columns present: {list(df.columns)}")

target_bpp = 0.4
target_psnr = 34.48

psnr_at_bpp_vals = []
bpp_at_psnr_vals = []

if key_col is not None:
    for _, g in df.groupby(key_col):
        g1 = g[[bpp_col, psnr_col]].dropna()
        psnr_v = interp_val(g1[bpp_col], g1[psnr_col], target_bpp)
        psnr_at_bpp_vals.append(psnr_v)
        g2 = g1.sort_values(psnr_col)
        bpp_v = interp_val(g2[psnr_col], g2[bpp_col], target_psnr)
        bpp_at_psnr_vals.append(bpp_v)
else:
    g1 = df[[bpp_col, psnr_col]].dropna()
    psnr_v = interp_val(g1[bpp_col], g1[psnr_col], target_bpp)
    bpp_v = interp_val(g1.sort_values(psnr_col)[psnr_col], g1.sort_values(psnr_col)[bpp_col], target_psnr)
    psnr_at_bpp_vals = [psnr_v] if psnr_v is not None else []
    bpp_at_psnr_vals = [bpp_v] if bpp_v is not None else []

psnr_med, psnr_std, n1 = summarize(psnr_at_bpp_vals)
bpp_med, bpp_std, n2 = summarize(bpp_at_psnr_vals)

print(f"Source: {used_path}")
print(f"Detected columns: key={key_col}, bpp={bpp_col}, psnr={psnr_col}")
if psnr_med is None:
    print(f"JPEG PSNR @ {target_bpp} bpp: insufficient coverage to interpolate")
else:
    print(f"JPEG PSNR @ {target_bpp} bpp: median={psnr_med:.3f}, std={psnr_std:.3f} (n={n1})")
if bpp_med is None:
    print(f"JPEG BPP @ {target_psnr:.0f} dB: insufficient coverage to interpolate")
else:
    print(f"JPEG BPP @ {target_psnr:.0f} dB: median={bpp_med:.3f}, std={bpp_std:.3f} (n={n2})")

out = {
    "source":[used_path],
    "key_col":[key_col],
    "bpp_col":[bpp_col],
    "psnr_col":[psnr_col],
    f"psnr@{target_bpp}_median":[psnr_med],
    f"psnr@{target_bpp}_std":[psnr_std],
    f"psnr@{target_bpp}_n":[n1],
    f"bpp@{int(target_psnr)}db_median":[bpp_med],
    f"bpp@{int(target_psnr)}db_std":[bpp_std],
    f"bpp@{int(target_psnr)}db_n":[n2],
}
pd.DataFrame(out).to_csv("plots/jpeg_medstd_summary.csv", index=False)
print("Saved: plots/jpeg_medstd_summary.csv")
