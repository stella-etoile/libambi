r"""One picture through JPEG, WebP, AVIF and AMBI: a rate-distortion graph, and a separate side-by-side picture of the telemetry.

  python scripts\codec_compare.py photo.png                        # INPUT = one picture, or a folder of pictures
  python scripts\codec_compare.py photo.png -o out\codecs --match psnr --lambda 0.03
  python scripts\codec_compare.py photo.png --crop 512 --lams 0.01 0.04 0.16 --qualities 10 30 50 70 90      # quick look

What it does, per picture:
  1. Sweeps every codec over its quality knob: JPEG, WebP and AVIF over --qualities (Pillow), AMBI over --lams (the real encoder, so
     this part takes by far the longest; every lambda is a full encode). Every file is really encoded and really decoded again; size,
     PSNR (RGB) and SSIM (Y) are measured on what comes out of the decoder.
  2. AMBI at --lambda (default 0.03) is the reference. JPEG, WebP and AVIF are each tuned (by searching their quality, 1..100) to
     --match size : the best quality whose file is no larger than AMBI's        (default; then compare PSNR / SSIM / how it looks), or
     --match psnr : the smallest file that reaches AMBI's PSNR                  (then compare the file sizes).
  3. Draws two pictures.

In the output folder (default codec_compare_out), for NAME = the picture's file name without extension:
  NAME.rd_curves.png    the graph: PSNR against bits per pixel, and SSIM(Y) against bits per pixel, one curve per codec. A ring marks the
                        point of each codec that the other picture compares. Zoomed to AMBI's range of bitrates.
  NAME.comparison.png   the telemetry side by side: the original, then JPEG, WebP, AVIF, AMBI. For each: the picture, the same detailed
                        crop at native pixels, and its numbers: file size, saved % vs the original file, bits per pixel, PSNR, SSIM(Y),
                        encode time, decode time. The best value of each row among the four codecs is highlighted.
  NAME.jpg  NAME.webp  NAME.avif  NAME.ambi    the compared files themselves
  NAME.telemetry.csv    every encode that was made (the sweeps and the search probes), with a column marking the compared ones

Needs matplotlib (pip install matplotlib). AVIF needs a Pillow that can write it (Pillow 11.3 or newer), or pip install pillow-avif-plugin;
without it AVIF is skipped and the rest still runs.

AMBI defaults are the working setup: profiles\clic-base-2k.ambiprofile, policies\clic2k_big.ambipolicy with the pruned exact search
(--prune-k 5), one worker process per physical core, the network on the GPU when there is one. --oracle uses the exact partitioner instead
(slow). --crop N cuts an N x N piece from the middle first, which keeps the AMBI sweeps short on big pictures.

Fair-play notes: JPEG is Pillow's (4:2:0 chroma, optimised Huffman tables); WebP is lossy, --webp-method 6 (slowest, smallest); AVIF is lossy,
--avif-speed 6. AMBI codes all three colour channels at full resolution. The times are for one encode of one picture: AMBI runs several
worker processes, the others one thread, so they are not a speed contest. Use a source that was never lossy compressed (PNG, TIFF, BMP):
a picture that was once a JPEG already carries JPEG's errors, which flatters JPEG (and WebP), and the script says so when you give one.
"""
import argparse, csv, io, os, pathlib, sys, textwrap, time
from dataclasses import dataclass, field
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import Image
from ambi_core import psnr_rgb
from ambi_core.bitstream import decode_image, encode_image
from ambi_core.metrics import ssim_y
from ambi_core.profile import Profile
from side_by_side import load_rgb

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
# LOSSY_SOURCES = (".jpg", ".jpeg", ".webp", ".avif", ".heic", ".heif")
LOSSY_SOURCES = (".jpg", ".jpeg", ".webp")
# the colours of the repo's other graphs (scripts\make_rd_plot.py): categorical slots 1-4, validated for adjacent pairs.
# Yellow and aqua are below 3:1 on the light surface, so every series is also named by a direct label, a legend and its own marker shape.
SURFACE, INK, INK2, GRID, PILL = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#e9e8e3"
COLORS = {"AMBI": "#2a78d6", "JPEG": "#eb6834", "WebP": "#1baf7a", "AVIF": "#eda100"}
MARKERS = {"AMBI": "o", "JPEG": "s", "WebP": "^", "AVIF": "D"}
# ORDER = ("JPEG", "WebP", "AVIF", "AMBI")
ORDER = ("JPEG", "WebP", "AMBI")
DPI = 130


def log(msg: str = "") -> None:
    print(msg, flush=True)


def rel(nbytes: float, ref: float) -> str:
    """'12.3% smaller' or '12.3% LARGER' than the reference."""
    d = 100 * (1 - nbytes / ref)
    return f"{abs(d):.1f}% smaller" if d >= 0 else f"{abs(d):.1f}% LARGER"


def secs(s: float) -> str:
    return f"{s:.2f} s" if s < 10 else f"{s:.1f} s"


def pick_device(spec: str) -> str:
    if spec != "auto":
        return spec
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def list_inputs(path: pathlib.Path):
    if path.is_dir():
        return [f for f in sorted(path.iterdir()) if f.is_file() and f.suffix.lower() in EXTS]
    return [path]


# ------------------------------------------------------------------------------------------------- codecs
@dataclass
class Run:
    codec: str
    param: float                 # quality (JPEG, WebP, AVIF) or lambda (AMBI)
    data: bytes
    bpp: float
    psnr: float
    ssim: float
    enc_s: float
    dec_s: float
    extra: dict = field(default_factory=dict)

    @property
    def bytes(self) -> int:
        return len(self.data)

    @property
    def setting(self) -> str:
        return f"lambda {self.param:g}" if self.codec == "AMBI" else f"quality {int(self.param)}"


class PillowCodec:
    """JPEG, WebP or AVIF through Pillow at an integer quality; every real encode is kept (cache), so searches never repeat one."""

    def __init__(self, name, fmt, ext, rgb, qrange, opts):
        self.name, self.fmt, self.ext, self.rgb, self.qrange, self.opts = name, fmt, ext, rgb, qrange, opts
        self.img = Image.fromarray(rgb)
        self.n_pix = rgb.shape[0] * rgb.shape[1]
        self.cache = {}

    def run(self, q) -> Run:
        q = int(q)
        if q in self.cache:
            return self.cache[q]
        buf = io.BytesIO()
        t0 = time.time()
        self.img.save(buf, self.fmt, quality=q, **self.opts)
        enc_s = time.time() - t0
        data = buf.getvalue()
        t0 = time.time()
        dec = self._decode(data)
        dec_s = time.time() - t0
        r = Run(self.name, q, data, len(data) * 8 / self.n_pix, psnr_rgb(self.rgb, dec), ssim_y(self.rgb, dec), enc_s, dec_s)
        self.cache[q] = r
        return r

    @staticmethod
    def _decode(data: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(data)) as im:
            return np.array(im.convert("RGB"))

    def decode(self, run: Run) -> np.ndarray:
        return self._decode(run.data)


class AmbiCodec:
    name, ext = "AMBI", ".ambi"

    def __init__(self, rgb, prof, part, how, mu=0.0):
        self.rgb, self.prof, self.part, self.how, self.mu = rgb, prof, part, how, mu
        self.n_pix = rgb.shape[0] * rgb.shape[1]
        self.cache = {}

    def warm_up(self) -> None:
        """One throw-away encode of a small corner, so starting the worker processes (and the network) is not counted in the first timing."""
        if self.part is None:
            return
        encode_image(np.ascontiguousarray(self.rgb[:64, :64]), self.prof, 0.03, self.mu, partitioner=self.part)

    def run(self, lam) -> Run:
        lam = float(lam)
        if lam in self.cache:
            return self.cache[lam]
        if hasattr(self.part, "reset_stats"):
            self.part.reset_stats()
        t0 = time.time()
        res = encode_image(self.rgb, self.prof, lam, self.mu, partitioner=self.part)
        enc_s = time.time() - t0
        pr = None
        if hasattr(self.part, "stats") and hasattr(self.part, "k"):
            st, nt, nr = self.part.stats, max(1, self.part.stats["tiles"]), self.part.space.n_rects
            pr = dict(k=self.part.k, looked=100 * st["looked_at"] / nt / nr, priced=100 * st["priced_as_leaf"] / nt / nr,
                      ms=1000 * st["seconds_wall"] / nt)
        data = res.data
        t0 = time.time()
        dec = decode_image(data, self.prof)
        dec_s = time.time() - t0
        r = Run("AMBI", lam, data, len(data) * 8 / self.n_pix, psnr_rgb(self.rgb, dec), ssim_y(self.rgb, dec), enc_s, dec_s,
                dict(deblock=res.deblock, prune=pr))
        self.cache[lam] = r
        return r

    def decode(self, run: Run) -> np.ndarray:
        return decode_image(run.data, self.prof)


def avif_problem():
    """None if this Pillow can write and read AVIF (natively or through a plugin), else a sentence saying what to install."""
    def probe():
        b = io.BytesIO()
        Image.new("RGB", (16, 16), (90, 120, 160)).save(b, "AVIF", quality=50)
        b.seek(0)
        Image.open(b).load()
    try:
        probe()
        return None
    except Exception:
        pass
    for mod in ("pillow_avif", "pillow_heif"):
        try:
            m = __import__(mod)
            if mod == "pillow_heif" and hasattr(m, "register_avif_opener"):
                m.register_avif_opener()
            probe()
            return None
        except Exception:
            pass
    return "this Pillow cannot write AVIF: upgrade Pillow (11.3 or newer) or run  pip install pillow-avif-plugin"


# ------------------------------------------------------------------------------------------ matching a quality
def largest_ok(ok, lo: int, hi: int, known) -> "int | None":
    """ok(x) is True up to some x* and False above it. The largest x in [lo, hi] with ok(x), or None if there is none. `known` are x values
    that have been evaluated already (testing them is free), which shrinks the bracket before the bisection starts."""
    a, b = lo - 1, hi + 1
    for x in sorted(k for k in known if lo <= k <= hi):
        if ok(x):
            a = max(a, x)
    for x in sorted(k for k in known if lo <= k <= hi):
        if x > a and not ok(x):
            b = x
            break
    while b - a > 1:
        m = (a + b) // 2
        if ok(m):
            a = m
        else:
            b = m
    return a if a >= lo else None


def match_codec(codec: PillowCodec, kind: str, ref: Run):
    """(Run, reached): the codec's quality tuned to AMBI's size (largest quality not bigger) or PSNR (smallest quality that reaches it)."""
    lo, hi = codec.qrange
    if kind == "size":
        q = largest_ok(lambda x: codec.run(x).bytes <= ref.bytes, lo, hi, list(codec.cache))
        return (codec.run(lo), False) if q is None else (codec.run(q), True)
    s = lo + hi                                           # mirror the axis: "smallest q with psnr >= target" is "largest r"
    r = largest_ok(lambda x: codec.run(s - x).psnr >= ref.psnr, lo, hi, [s - q for q in codec.cache])
    return (codec.run(hi), False) if r is None else (codec.run(s - r), True)


# ------------------------------------------------------------------------------------------------- the pictures
def pick_box(rgb: np.ndarray, size: int):
    """(x, y, s): a detailed but not extreme square window (the one at the 90th percentile of local gradient energy), the same for every
    codec, so the zoom shows what the codecs do to real detail without being the single hardest spot of the picture."""
    H, W = rgb.shape[:2]
    s = int(min(size, H, W))
    g = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    e = np.abs(np.diff(g, axis=0, prepend=g[:1])) + np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    ii = np.pad(e.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    step = max(1, s // 4)
    ys, xs = np.arange(0, H - s + 1, step), np.arange(0, W - s + 1, step)
    S = ii[np.ix_(ys + s, xs + s)] - ii[np.ix_(ys, xs + s)] - ii[np.ix_(ys + s, xs)] + ii[np.ix_(ys, xs)]
    k = int(np.argmin(np.abs(S - np.percentile(S, 90))))
    return int(xs[k % len(xs)]), int(ys[k // len(xs)]), s


def thumb(arr: np.ndarray, max_side: int = 1500):
    """(display array, factor): a big picture is scaled down once for display (Lanczos), the crops use the full-size array."""
    H, W = arr.shape[:2]
    f = min(1.0, max_side / max(H, W))
    if f >= 1.0:
        return arr, 1.0
    lz = getattr(Image, "Resampling", Image).LANCZOS
    return np.array(Image.fromarray(arr).resize((max(1, round(W * f)), max(1, round(H * f))), lz)), f


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def draw_rd(path, title, sub, series, compared) -> None:
    """series: {codec: [Run]}; compared: {codec: Run}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    lw = 2 * 72 / DPI * 1.0
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), dpi=DPI, facecolor=SURFACE)
    a_bpp = [r.bpp for r in series["AMBI"]]
    xlo, xhi = min(a_bpp) * 0.8, max(a_bpp) * 1.28
    for ax, key, ylabel in zip(axes, ("psnr", "ssim"), ("PSNR (dB), RGB — higher is better", "SSIM on luma (Y) — higher is better")):
        style_axes(ax)
        curves, ys = {}, []
        for name in ORDER:
            if name not in series:
                continue
            runs = sorted((r for r in series[name] if np.isfinite(r.psnr) and np.isfinite(r.ssim)), key=lambda r: r.bpp)
            if not runs:
                continue
            x, y = np.array([r.bpp for r in runs]), np.array([getattr(r, key) for r in runs])
            curves[name] = (x, y)
            vis = (x >= xlo) & (x <= xhi)
            ys += list(y[vis])
        for name, (x, y) in curves.items():
            ax.plot(x, y, color=COLORS[name], lw=lw, solid_capstyle="round", solid_joinstyle="round", marker=MARKERS[name], ms=5.4,
                    mec=SURFACE, mew=1.0, zorder=3 if name == "AMBI" else 2)
        for name, r in compared.items():
            if xlo <= r.bpp <= xhi:
                ys.append(getattr(r, key))
        ylo, yhi = min(ys), max(ys)
        pad = (yhi - ylo) * 0.06 or 0.5
        ylo, yhi = ylo - pad, yhi + pad
        if key == "ssim":
            yhi = min(yhi, 1.0005)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        for name, r in compared.items():                  # the rings: the files the other picture compares
            if xlo <= r.bpp <= xhi:
                ax.scatter([r.bpp], [getattr(r, key)], s=150, facecolors="none", edgecolors=INK, linewidths=1.5, zorder=6)
        # direct labels. A curve that leaves through the top is named just above the frame at the spot where it leaves; every other curve is
        # named at its end (the right edge or its last point), pushed apart vertically from labels that are close to it sideways
        xr, yr = xhi - xlo, yhi - ylo
        side, top = [], []
        for name, (x, y) in curves.items():
            x0_, x1_ = max(xlo, float(x.min())), min(xhi, float(x.max()))
            if x0_ > x1_:
                continue
            grid = np.linspace(x0_, x1_, 600)
            gy = np.interp(grid, x, y)
            seen = np.flatnonzero((gy >= ylo) & (gy <= yhi))
            if not len(seen):
                continue
            k = seen[-1]
            if k < len(grid) - 1 and gy[k + 1] > yhi:
                top.append([name, float(grid[k])])
            else:
                side.append([name, float(grid[k]), float(gy[k])])
        half = 0.035 * yr
        for t in side:
            t[2] = min(max(t[2], ylo + half), yhi - half)
        side.sort(key=lambda t: t[2])
        near = 0.16 * xr
        for i in range(len(side)):                        # push up away from closer-by labels below
            for j in range(i):
                if abs(side[i][1] - side[j][1]) < near:
                    side[i][2] = max(side[i][2], side[j][2] + 2 * half)
        for i in range(len(side) - 1, -1, -1):            # whatever went over the top is pushed back down
            side[i][2] = min(side[i][2], yhi - half)
            for j in range(i + 1, len(side)):
                if abs(side[i][1] - side[j][1]) < near:
                    side[i][2] = min(side[i][2], side[j][2] - 2 * half)
        for name, lx, ly in side:
            ax.text(lx + 0.012 * xr, ly, name, color=INK2, fontsize=9.5, va="center", ha="left", zorder=5,
                    path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)], clip_on=False)
        top.sort(key=lambda t: t[1])
        sep = 0.085 * xr                                   # about one label's width
        for i in range(1, len(top)):
            top[i][1] = max(top[i][1], top[i - 1][1] + sep)
        over = top[-1][1] - (xhi - 0.04 * xr) if top else 0
        if over > 0:
            for t in top:
                t[1] -= over
        for name, lx in top:
            ax.text(lx, yhi + 0.012 * yr, name, color=INK2, fontsize=9.5, va="bottom", ha="center", zorder=5, clip_on=False)
        ax.set_xlabel("bits per pixel (file bytes x 8 / pixels) — smaller is better", color=INK2, fontsize=9.5)
        ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    handles = [plt.Line2D([0], [0], color=COLORS[n], lw=lw, marker=MARKERS[n], ms=5.4, mec=SURFACE, label=n) for n in ORDER if n in series]
    handles.append(plt.Line2D([0], [0], color="none", marker="o", ms=10, mfc="none", mec=INK, mew=1.5, label="compared in the side-by-side picture"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9.5, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.012, 0.965, title, color=INK, fontsize=14, fontweight="bold", ha="left", va="center")
    fig.text(0.012, 0.915, sub, color=INK2, fontsize=9.5, ha="left", va="center")
    fig.subplots_adjust(left=0.07, right=0.93, top=0.86, bottom=0.2, wspace=0.24)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


BEST = {"size": (min, 0.01, True), "psnr": (max, 0.05, False), "ssim": (max, 0.0005, False), "enc": (min, 0.10, True), "dec": (min, 0.10, True)}


def draw_comparison(path, title, sub, foot, cols, box, orig_rgb) -> None:
    """cols: list of dicts (name, setting, pix, rows [(label, value, key, raw)], notes [str]); the first is the original."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    H, W = orig_rgb.shape[:2]
    n = len(cols)
    cw = 3.4
    h1, h2, h3 = cw * min(max(H / W, 0.55), 1.3), cw, 3.0
    top, bot = 1.0, 0.95
    fig = plt.figure(figsize=(cw * n + 0.5, top + h1 + h2 + h3 + bot + 0.7), dpi=DPI, facecolor=SURFACE)
    gs = fig.add_gridspec(3, n, height_ratios=[h1, h2, h3], left=0.25 / (cw * n + 0.5), right=1 - 0.25 / (cw * n + 0.5),
                          top=1 - (top + 0.25) / fig.get_figheight(), bottom=bot / fig.get_figheight(), hspace=0.16, wspace=0.07)
    x0, y0, s = box
    # the best value of each row among the lossy codecs (ties within a tolerance are all marked)
    best = {}
    for key, (fn, tol, rel_tol) in BEST.items():
        vals = {i: c["raw"][key] for i, c in enumerate(cols) if c.get("lossy") and key in c["raw"]}
        if len(vals) < 2:
            continue
        b = fn(vals.values())
        winners = {i for i, v in vals.items() if abs(v - b) <= (tol * abs(b) if rel_tol else tol)}
        if key in ("enc", "dec"):                           # times are compared as printed: 0.00 s against 0.00 s is a tie, not a win,
            b = fn(round(v, 2) for v in vals.values())      # and a time is marked only when one codec is alone in front
            winners = {i for i, v in vals.items() if round(v, 2) == b}
            if len(winners) > 1:
                continue
        if len(winners) <= len(vals) - 2:                   # a row where nearly everything ties has no winner worth marking
            best[key] = winners
    for i, c in enumerate(cols):
        disp, f = thumb(c["pix"])
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(disp, interpolation="antialiased")
        for lw_, col in ((3.4, "white"), (1.4, "#0b0b0b")):
            ax.add_patch(Rectangle((x0 * f - 0.5, y0 * f - 0.5), s * f, s * f, fill=False, ec=col, lw=lw_))
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.add_patch(Rectangle((0, 1.012), 1, 0.014, transform=ax.transAxes, color=c["color"], clip_on=False))
        ax.text(0, 1.06, c["name"], transform=ax.transAxes, color=INK, fontsize=13, fontweight="bold", ha="left", va="bottom")
        ax.text(1, 1.06, c["setting"].replace("lambda", "\u03bb"), transform=ax.transAxes, color=INK2, fontsize=10, ha="right", va="bottom")
        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(c["pix"][y0:y0 + s, x0:x0 + s], interpolation="nearest")
        ax2.set_xticks([]), ax2.set_yticks([])
        for sp in ax2.spines.values():
            sp.set_visible(False)
        if i == 0:
            ax2.text(0.02, 0.02, f"detail at native pixels (the box above)", transform=ax2.transAxes, color="white", fontsize=8.5, ha="left",
                     va="bottom", bbox=dict(boxstyle="round,pad=0.25", fc="#0b0b0bb0", ec="none"))
        ax3 = fig.add_subplot(gs[2, i])
        ax3.set_xlim(0, 1)
        nrows = 12
        ax3.set_ylim(nrows, 0)
        ax3.axis("off")
        for r, (label, value, key) in enumerate(c["rows"]):
            y = r + 0.5
            ax3.text(0.0, y, label, color=INK2, fontsize=10.5, ha="left", va="center")
            is_best = key is not None and i in best.get(key, ())
            ax3.text(1.0, y, value, color=INK, fontsize=11, ha="right", va="center", fontweight="bold" if is_best else "normal",
                     bbox=dict(boxstyle="round,pad=0.22", fc=PILL, ec="none") if is_best else None)
        lines = [ln for note in c["notes"] for ln in textwrap.wrap(note, 44)][:7]
        for k, ln in enumerate(lines):
            ax3.text(0.0, 7.5 + k * 0.62, ln, color=INK2, fontsize=8.6, ha="left", va="center", style="italic")
    fig.text(0.25 / (cw * n + 0.5), 1 - 0.32 / fig.get_figheight(), title, color=INK, fontsize=16, fontweight="bold", ha="left", va="center")
    fig.text(0.25 / (cw * n + 0.5), 1 - 0.68 / fig.get_figheight(), sub, color=INK2, fontsize=10.5, ha="left", va="center")
    fig.text(0.25 / (cw * n + 0.5), 0.14 / fig.get_figheight(), foot, color=INK2, fontsize=8.8, ha="left", va="bottom", linespacing=1.5)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------- one picture
def column_rows(run: Run, orig_bytes: int):
    return [("file size", f"{run.bytes:,} B", "size"), ("vs original file", rel(run.bytes, orig_bytes), None),
            ("bitrate", f"{run.bpp:.3f} bpp", None), ("PSNR (RGB)", f"{run.psnr:.2f} dB", "psnr"), ("SSIM (Y)", f"{run.ssim:.4f}", "ssim"),
            ("encode time", secs(run.enc_s), "enc"), ("decode time", secs(run.dec_s), "dec")]


def process(f: pathlib.Path, a, prof, part, how, avif_err, out_dir: pathlib.Path) -> None:
    rgb, note = load_rgb(f)
    cropped = bool(a.crop) and a.crop < max(rgb.shape[:2])
    if cropped:
        H0, W0 = rgb.shape[:2]
        ch, cwid = min(a.crop, H0), min(a.crop, W0)
        y, x = (H0 - ch) // 2, (W0 - cwid) // 2
        rgb = np.ascontiguousarray(rgb[y:y + ch, x:x + cwid])
    H, W = rgb.shape[:2]
    stem = f.stem
    if cropped:                                             # a crop has no file of its own: the lossless PNG of the crop is the reference
        b = io.BytesIO()
        Image.fromarray(rgb).save(b, "PNG", optimize=True)
        orig_bytes, fmt = b.tell(), "PNG of the crop"
    else:
        orig_bytes, fmt = f.stat().st_size, f.suffix.lstrip(".").upper()
    raw = H * W * 3
    log(f"{f.name}   {W}x{H}   {orig_bytes:,} bytes ({fmt}) · raw RGB {raw:,} bytes" + (f"   [{note}]" if note else "")
        + (f"   [centre {a.crop}x{a.crop} crop]" if cropped else ""))
    if f.suffix.lower() in LOSSY_SOURCES:
        log(f"  WARNING: {f.name} is already lossy ({f.suffix.lstrip('.').upper()}); its errors are part of the 'original' and flatter JPEG and WebP. "
            f"Use a PNG/TIFF/BMP source for a fair comparison.")

    pil = [PillowCodec("JPEG", "JPEG", ".jpg", rgb, (1, 95), dict(optimize=True))]
    if max(H, W) <= 16383:
        pil.append(PillowCodec("WebP", "WEBP", ".webp", rgb, (1, 100), dict(method=a.webp_method)))
    else:
        log("  WebP skipped: WebP cannot hold a side longer than 16383 px")
    if avif_err is None:
        pil.append(PillowCodec("AVIF", "AVIF", ".avif", rgb, (1, 100), dict(speed=a.avif_speed)))
    ambi = AmbiCodec(rgb, prof, part, how, a.mu)

    # 1. the sweeps
    for c in pil:
        log(f"  {c.name}: sweeping quality {', '.join(map(str, a.qualities))}")
        for q in a.qualities:
            if c.qrange[0] <= q <= c.qrange[1]:
                r = c.run(q)
                log(f"     quality {q:>3}  {r.bytes:>10,} B  {r.bpp:6.3f} bpp  PSNR {r.psnr:6.2f}  SSIM {r.ssim:.4f}  enc {secs(r.enc_s)}")
    lams = sorted(set(a.lams) | {a.lam})
    log(f"  AMBI: {how}")
    ambi.warm_up()
    for lam in lams:
        r = ambi.run(lam)
        log(f"     lambda {lam:<6g} {r.bytes:>10,} B  {r.bpp:6.3f} bpp  PSNR {r.psnr:6.2f}  SSIM {r.ssim:.4f}  enc {secs(r.enc_s)}"
            f"{'   <- compared' if lam == a.lam else ''}")
    ref = ambi.run(a.lam)

    # 2. match JPEG / WebP / AVIF to the AMBI file
    compared, reached = {"AMBI": ref}, {"AMBI": True}
    log(f"  matching each codec to AMBI's {'file size' if a.match == 'size' else 'PSNR'} "
        f"({ref.bytes:,} B, {ref.psnr:.2f} dB)")
    for c in pil:
        r, ok = match_codec(c, a.match, ref)
        compared[c.name], reached[c.name] = r, ok
        log(f"     {c.name:<5} quality {int(r.param):>3}  {r.bytes:>10,} B  PSNR {r.psnr:6.2f}  SSIM {r.ssim:.4f}"
            + ("" if ok else "   (the target cannot be reached)"))

    # 3. files, csv, pictures
    codecs = {c.name: c for c in pil}
    codecs["AMBI"] = ambi
    for name, r in compared.items():
        (out_dir / f"{stem}{codecs[name].ext}").write_bytes(r.data)
    with open(out_dir / f"{stem}.telemetry.csv", "w", newline="", encoding="utf-8") as cf:
        cw = csv.writer(cf)
        cw.writerow(["codec", "setting", "param", "bytes", "bpp", "pct_smaller_than_original_file", "psnr_rgb_db", "ssim_y", "encode_s",
                     "decode_s", "compared"])
        for name in ORDER:
            if name in codecs:
                for k in sorted(codecs[name].cache):
                    r = codecs[name].cache[k]
                    cw.writerow([name, r.setting, k, r.bytes, f"{r.bpp:.4f}", f"{100 * (1 - r.bytes / orig_bytes):.2f}", f"{r.psnr:.3f}",
                                 f"{r.ssim:.5f}", f"{r.enc_s:.3f}", f"{r.dec_s:.3f}", int(compared.get(name) is r)])

    series = {n: list(codecs[n].cache.values()) for n in ORDER if n in codecs}
    rd_path, cmp_path = out_dir / f"{stem}.rd_curves.png", out_dir / f"{stem}.comparison.png"
    lossy_src = f.suffix.lower() in LOSSY_SOURCES
    draw_rd(rd_path, f"{f.name}: size against quality for JPEG, WebP, AVIF and AMBI",
            f"{W} x {H} px · original file {orig_bytes:,} B ({orig_bytes * 8 / (W * H):.2f} bpp) · "
            f"zoomed to AMBI's range · PSNR on RGB, SSIM on luma · every point is a real file decoded again"
            + (" · SOURCE IS ALREADY LOSSY" if lossy_src else ""), series, compared)

    xb, yb, sb = (a.box[0], a.box[1], min(a.zoom_size, H, W)) if a.box else pick_box(rgb, a.zoom_size)
    xb, yb = max(0, min(xb, W - sb)), max(0, min(yb, H - sb))
    cols = [dict(name="Original", setting=fmt, pix=rgb, color="#8a8984", lossy=False, raw={}, notes=[
        "the reference for PSNR and SSIM" if not lossy_src else "already lossy: not a clean reference"],
        rows=[("file size", f"{orig_bytes:,} B", None), ("vs original file", "reference", None), ("bitrate", f"{orig_bytes * 8 / (W * H):.3f} bpp", None),
              ("PSNR (RGB)", "lossless", None), ("SSIM (Y)", "1.0000", None), ("pixels", f"{W} x {H}", None), ("raw RGB", f"{raw:,} B", None)])]
    for name in ORDER:
        if name not in compared:
            continue
        r = compared[name]
        notes = []
        if name == "AMBI":
            notes.append(how)
            pr = r.extra.get("prune")
            if pr:
                notes.append(f"pruning looked at {pr['looked']:.1f}% of rectangles, {pr['ms']:.0f} ms per tile")
            notes.append(f"deblock level {r.extra.get('deblock', 0)} · {codecs['AMBI'].prof.name}")
        elif not reached[name]:
            if a.match == "size":
                notes.append(f"cannot get as small as AMBI: its smallest is {r.bytes / ref.bytes:.2f}x AMBI's size")
            else:
                notes.append(f"cannot reach AMBI's PSNR: best is {r.psnr:.2f} dB at quality {int(r.param)}")
        cols.append(dict(name=name, setting=r.setting, pix=codecs[name].decode(r), color=COLORS[name], lossy=True,
                         raw=dict(size=r.bytes, psnr=r.psnr, ssim=r.ssim, enc=r.enc_s, dec=r.dec_s), notes=notes, rows=column_rows(r, orig_bytes)))
    if a.match == "size":
        what = f"JPEG, WebP and AVIF quality searched so each file is no larger than AMBI's {ref.bytes:,} B"
    else:
        what = f"JPEG, WebP and AVIF quality searched so each reaches AMBI's PSNR of {ref.psnr:.2f} dB with the smallest file"
    foot = (f"AMBI at lambda {a.lam:g}; {what}. PSNR on RGB, SSIM on luma, both measured on the decoded files. Bold = best of the four in that row"
            f" (ties within a small tolerance).\nJPEG 4:2:0, optimised tables · WebP lossy, method {a.webp_method} · AVIF lossy, speed {a.avif_speed} · AMBI codes all"
            f" three channels at full resolution. Times are one encode of one picture; AMBI uses worker processes, the others one thread."
            + ("\nSOURCE IS ALREADY LOSSY: its errors are part of the original, which flatters JPEG and WebP." if lossy_src else ""))
    if avif_err:
        foot += f"\nAVIF skipped: {avif_err}"
    draw_comparison(cmp_path, f"{f.name}: the same picture, four codecs", f"{W} x {H} px · compared at AMBI's {'file size' if a.match == 'size' else 'PSNR'}"
                    f" (lambda {a.lam:g}) · the detail row is the boxed {sb} x {sb} px area, at native pixels", foot, cols, (xb, yb, sb), rgb)
    log(f"  wrote {rd_path.name}, {cmp_path.name}, {stem}.telemetry.csv and the compared files\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="one picture through JPEG, WebP, AVIF and AMBI: a rate-distortion graph and a side-by-side telemetry picture")
    ap.add_argument("input", help="a picture, or a folder of pictures")
    ap.add_argument("-o", "--out-dir", default="codec_compare_out")
    ap.add_argument("--match", choices=("size", "psnr"), default="size",
                    help="how JPEG, WebP and AVIF are matched to AMBI in the side-by-side picture: equal file size (default) or equal PSNR")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.03, help="AMBI's lambda for the compared picture (higher = smaller file), default 0.03")
    ap.add_argument("--lams", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32], metavar="L",
                    help="AMBI lambdas for the graph (each one is a full encode; default 0.005 ... 0.32)")
    ap.add_argument("--qualities", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95], metavar="Q",
                    help="JPEG / WebP / AVIF qualities for the graph (default 5 10 20 ... 90 95)")
    ap.add_argument("--crop", type=int, default=0, metavar="N", help="use only an N x N piece from the middle of the picture (quick runs on big pictures)")
    ap.add_argument("--zoom-size", type=int, default=96, metavar="PX", help="side of the detail window in the side-by-side picture (default 96)")
    ap.add_argument("--box", type=int, nargs=2, metavar=("X", "Y"), help="top-left corner of the detail window (default: a detailed area picked automatically)")
    ap.add_argument("--webp-method", type=int, default=6, help="WebP effort 0-6 (default 6, the slowest and smallest)")
    ap.add_argument("--avif-speed", type=int, default=6, help="AVIF speed 0-10 (default 6; lower = slower and smaller)")
    ap.add_argument("--profile", default=os.path.join("profiles", "clic-base-2k.ambiprofile"))
    ap.add_argument("--policy", default=os.path.join("policies", "clic2k_big.ambipolicy"), help="trained .ambipolicy (the partitioner)")
    ap.add_argument("--oracle", action="store_true", help="use the exact oracle instead of the policy (slow)")
    ap.add_argument("--prune-k", type=int, default=5, metavar="K",
                    help="exact search over the policy's K best actions per rectangle (default 5; 0 = the policy alone)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), metavar="N",
                    help="processes that price the candidate leaves (default: half the logical cores = the physical cores)")
    ap.add_argument("--device", default="auto", help="where the policy network runs: auto, cpu, cuda (default auto)")
    ap.add_argument("--mu", type=float, default=0.0, help="extra per-block bit cost for AMBI (bias toward larger blocks)")
    a = ap.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        sys.exit("matplotlib is needed for the pictures: pip install matplotlib")
    files = list_inputs(pathlib.Path(a.input))
    if not files:
        sys.exit(f"no pictures found in {a.input}")
    if not pathlib.Path(a.profile).exists():
        sys.exit(f"profile not found: {a.profile}  (give --profile FILE)")
    prof = Profile.load(a.profile)

    part, how = None, "exact oracle"
    if not a.oracle:
        if not pathlib.Path(a.policy).exists():
            sys.exit(f"policy not found: {a.policy}  (give --policy FILE, or --oracle for the slow exact partitioner)")
        from ambi_rl.pruned import make_partitioner
        device = pick_device(a.device)
        workers = a.workers if a.prune_k else 1
        part = make_partitioner(a.policy, prof, device, a.prune_k, workers)
        how = (f"policy {pathlib.Path(a.policy).name} + exact search over its top {a.prune_k} actions, {workers} worker(s), {device}"
               if a.prune_k else f"policy {pathlib.Path(a.policy).name} alone, {device}")
    avif_err = avif_problem()
    if avif_err:
        log(f"AVIF will be skipped: {avif_err}")
    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"profile {prof.name} (id {prof.profile_id}), AMBI lambda {a.lam:g}, matched at equal {a.match}\noutput folder: {out_dir.resolve()}\n")
    try:
        for f in files:
            process(f, a, prof, part, how, avif_err, out_dir)
    finally:
        if part is not None and hasattr(part, "close"):
            part.close()
    return 0


# everything above only defines things: the pruner's worker processes (spawn) import this file again and must not re-run the script
if __name__ == "__main__":
    raise SystemExit(main())
