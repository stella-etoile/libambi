from __future__ import annotations

import io
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ambi_core.bitstream import decode_image, encode_image, read_info
from ambi_core.color import psnr_rgb
from ambi_core.errors import AmbiFormatError
from ambi_core.metrics import ssim_y
from ambi_core.profile import Profile
from ambi_core.search import encode_to_target

LAM_MIN, LAM_MAX = 0.005, 0.32
DEFAULT_LAM = 0.03
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

PRUNE_PRESETS = [
    ("Policy only", 0, "fastest · files roughly 10-17% bigger than the exact search"),
    ("k = 3", 3, "fast · files roughly 3-5% bigger than the exact search"),
    ("k = 5  (recommended)", 5, "good balance · files roughly 1-4% bigger than the exact search"),
    ("k = 8", 8, "slower · closer to exact (file size cost not measured)"),
    ("Exact oracle", -1, "very slow, about 1.4 s per 64x64 tile · the best AMBI can do, no policy needed"),
]


class EngineError(Exception):
    """Something the user can fix (missing file, wrong profile...). The message is meant to be shown."""


class Cancelled(Exception):
    pass


def quality_to_lam(q: float) -> float:
    t = min(100.0, max(0.0, float(q))) / 100.0
    return float(math.exp((1 - t) * math.log(LAM_MAX) + t * math.log(LAM_MIN)))


def lam_to_quality(lam: float) -> float:
    return 100.0 * (math.log(LAM_MAX) - math.log(lam)) / (math.log(LAM_MAX) - math.log(LAM_MIN))


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


def torch_status() -> tuple[bool, bool]:
    try:
        import torch
    except ImportError:
        return False, False
    return True, bool(torch.cuda.is_available())


def resolve_device(spec: str) -> str:
    if spec == "auto":
        return "cuda" if torch_status()[1] else "cpu"
    return spec


def load_picture(path) -> tuple[np.ndarray, str | None]:
    im = ImageOps.exif_transpose(Image.open(path))
    note = None
    if im.mode == "F":
        arr = np.asarray(im, dtype=np.float64)
        arr = arr * 255.0 if arr.max() <= 1.0 else arr
        im, note = Image.fromarray(np.clip(np.nan_to_num(arr).round(), 0, 255).astype(np.uint8), "L"), "float image scaled to 8-bit"
    if im.mode in ("I;16", "I;16L", "I;16B", "I"):
        arr = np.asarray(im, dtype=np.float64)
        if arr.max() > 255:
            arr = arr / 65535.0 * 255.0
        im, note = Image.fromarray(np.clip(arr.round(), 0, 255).astype(np.uint8), "L"), "16-bit scaled to 8-bit"
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        if rgba.getchannel("A").getextrema()[0] < 255:
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            bg.alpha_composite(rgba)
            im, note = bg, "transparency flattened onto white"
    return np.array(im.convert("RGB")), note


def png_size(rgb: np.ndarray) -> int:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG", optimize=True, compress_level=9)
    return len(buf.getvalue())


def jpeg_at_psnr(rgb: np.ndarray, target_psnr: float) -> dict:
    def trial(q):
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "JPEG", quality=q, subsampling="4:2:0", optimize=True)
        data = buf.getvalue()
        return len(data), psnr_rgb(rgb, np.array(Image.open(io.BytesIO(data)).convert("RGB")))
    n, p = trial(100)
    if p < target_psnr:
        return dict(quality=100, bytes=n, psnr=p, reached=False)
    best = (100, n, p)
    lo, hi = 1, 100
    while lo < hi:
        mid = (lo + hi) // 2
        n, p = trial(mid)
        if p >= target_psnr:
            best, hi = (mid, n, p), mid
        else:
            lo = mid + 1
    return dict(quality=best[0], bytes=best[1], psnr=best[2], reached=True)


def _over_message(nbytes: int, target: int, probes: int) -> str:
    over = nbytes - target
    if over <= 0.02 * target:
        return (f"closest file after {probes} encodes: {nbytes:,} bytes, {over:,} bytes ({100 * over / target:.2f}%) over the target; "
                f"file size moves in steps here, so no λ lands exactly under it")
    return (f"could not get under the target: the smallest file the search found is {nbytes:,} bytes, {100 * over / target:.0f}% over it "
            f"(λ is limited to {LAM_MIN}..{LAM_MAX})")


def solve_lambda(measure, kind: str, target: float, lam0: float = DEFAULT_LAM, max_probes: int = 7,
                 lam_min: float = LAM_MIN, lam_max: float = LAM_MAX):
    size = kind == "size"
    if kind not in ("size", "psnr"):
        raise ValueError(kind)

    def h(v):
        return math.log(max(v, 1.0) / target) if size else v - target

    window = (math.log(0.97), 0.0) if size else (0.0, 0.10)
    slope_guess = -0.33 if size else -1.5
    lo_x, hi_x = math.log(lam_min), math.log(lam_max)
    x = min(max(math.log(lam0), lo_x), hi_x)
    pts = []
    for _ in range(max_probes):
        lam = math.exp(x)
        v = measure(lam)
        hh = h(v)
        pts.append((x, hh, lam, v))
        if window[0] <= hh <= window[1]:
            break
        left = [p for p in pts if p[1] > 0]
        right = [p for p in pts if p[1] < 0]
        if left and right:
            a, b = max(left, key=lambda p: p[0]), min(right, key=lambda p: p[0])
            if b[0] - a[0] < 0.04:
                break
            t = min(0.9, max(0.1, a[1] / (a[1] - b[1])))
            x_new = a[0] + t * (b[0] - a[0])
        else:
            p, slope = pts[-1], slope_guess
            if len(pts) >= 2 and pts[-1][0] != pts[-2][0]:
                s_emp = (pts[-1][1] - pts[-2][1]) / (pts[-1][0] - pts[-2][0])
                if slope_guess * 3 <= s_emp <= slope_guess / 3:
                    slope = s_emp
            x_new = p[0] + max(-1.4, min(1.4, -p[1] / slope))
        x_new = min(max(x_new, lo_x), hi_x)
        if any(abs(x_new - q[0]) < 1e-3 for q in pts):
            break
        x = x_new
    feasible = [i for i, p in enumerate(pts) if (p[1] <= 0 if size else p[1] >= 0)]
    if feasible:
        best = (max if size else min)(feasible, key=lambda i: pts[i][1])
        reached = True
    else:
        best = (min if size else max)(range(len(pts)), key=lambda i: pts[i][1])
        reached = False
    return [(p[2], p[3]) for p in pts], best, reached


@dataclass
class EncodeSettings:
    profile_path: str
    policy_path: str = ""
    prune_k: int = 5
    workers: int = 1
    device: str = "auto"
    mode: str = "quality"
    lam: float = DEFAULT_LAM
    target_bytes: int = 0
    target_psnr: float = 36.0
    compare: bool = True
    mu: float = 0.0


@dataclass
class EncodeOutcome:
    data: bytes
    lam: float
    width: int
    height: int
    psnr: float
    ssim: float
    decoded: np.ndarray
    probes: list
    reached: bool
    message: str
    seconds: float
    encode_seconds: float
    decode_seconds: float
    deblock: int
    prune: dict | None
    mode: str
    how: str
    png_bytes: int | None = None
    jpeg: dict | None = None
    original_file_bytes: int | None = None

    @property
    def nbytes(self) -> int:
        return len(self.data)


@dataclass
class DecodeOutcome:
    rgb: np.ndarray
    info: dict
    profile_name: str
    seconds: float


class _Progress:
    def __init__(self, on_progress, cancel, expected: int):
        self.on_progress, self.cancel, self.expected = on_progress, cancel, max(1, expected)
        self.probe, self.label = 0, ""

    def check(self):
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled()

    def start_probe(self, index: int, label: str):
        self.check()
        self.probe, self.label = index, label
        if self.on_progress:
            self.on_progress(label, min(0.99, index / self.expected))

    def phase(self, name: str, lo: float, hi: float):
        def cb(done, total):
            self.check()
            if self.on_progress:
                f = lo + (hi - lo) * done / max(1, total)
                self.on_progress(f"{self.label} · {name} {done}/{total} tiles", min(0.99, (self.probe + f) / self.expected))
        return cb


class Engine:
    def __init__(self):
        self._profiles: dict = {}
        self._part = None
        self._part_key = None

    def profile(self, path) -> Profile:
        p = Path(path)
        if not p.is_file():
            raise EngineError(f"profile not found: {p}")
        mtime = p.stat().st_mtime
        hit = self._profiles.get(str(p))
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            prof = Profile.load(p)
        except Exception as e:
            raise EngineError(f"could not read the profile {p.name}: {e}") from None
        self._profiles[str(p)] = (mtime, prof)
        return prof

    def profile_for(self, data: bytes, dirs) -> Profile:
        info = read_info(data)
        same_id = []
        for d in dirs:
            if d and Path(d).is_dir():
                for f in sorted(Path(d).glob("*.ambiprofile")):
                    try:
                        p = self.profile(f)
                    except EngineError:
                        continue
                    if p.profile_id == info["profile_id"]:
                        if f"{p.fingerprint:08x}" == info["fingerprint"]:
                            return p
                        same_id.append(p.name)
        if same_id:
            raise EngineError(f"the file was made with profile id {info['profile_id']} (fingerprint {info['fingerprint']}); "
                              f"{', '.join(same_id)} has that id but different tables, so it cannot decode this file")
        raise EngineError(f"no profile with id {info['profile_id']} found; put its .ambiprofile in the profile folder "
                          f"({', '.join(str(d) for d in dirs if d)})")

    def _partitioner(self, prof: Profile, s: EncodeSettings):
        if s.prune_k < 0:
            return None
        if not s.policy_path or not Path(s.policy_path).is_file():
            raise EngineError("choose a policy file (.ambipolicy), or pick 'Exact oracle' for the slow search that needs none")
        have_torch, _ = torch_status()
        if not have_torch:
            raise EngineError("the policy needs PyTorch (pip install torch); 'Exact oracle' works without it")
        device = resolve_device(s.device)
        workers = s.workers if s.prune_k > 0 else 1
        key = (prof.fingerprint, str(Path(s.policy_path).resolve()), Path(s.policy_path).stat().st_mtime, s.prune_k, workers, device)
        if key == self._part_key and self._part is not None:
            return self._part
        self._drop_partitioner()
        from ambi_rl.pruned import make_partitioner
        try:
            self._part = make_partitioner(s.policy_path, prof, device, s.prune_k, workers)
        except Exception as e:
            raise EngineError(f"could not load the policy {Path(s.policy_path).name} with this profile: {e}") from None
        self._part_key = key
        return self._part

    def _drop_partitioner(self):
        part, self._part, self._part_key = self._part, None, None
        if part is not None and hasattr(part, "close"):
            part.close()

    def close(self):
        """Stop the worker processes of the pruned search."""
        self._drop_partitioner()

    def encode(self, rgb: np.ndarray, s: EncodeSettings, on_progress=None, cancel=None,
               original_file_bytes: int | None = None) -> EncodeOutcome:
        t_all = time.time()
        H, W = rgb.shape[:2]
        if s.mode == "size" and s.target_bytes <= 0:
            raise EngineError("enter the file size to aim for")
        prof = self.profile(s.profile_path)
        part = self._partitioner(prof, s)
        pruned = part is not None and hasattr(part, "k")
        how = ("exact oracle" if part is None else f"policy {Path(s.policy_path).name} alone" if not pruned
               else f"policy {Path(s.policy_path).name} + exact search over its top {s.prune_k} actions")
        searching = s.mode in ("size", "psnr")
        prog = _Progress(on_progress, cancel, 5 if searching and part is not None else 1)
        try:
            if part is None and searching:
                out = self._encode_oracle_target(rgb, prof, s, prog)
            else:
                out = self._encode_policy(rgb, prof, part, pruned, s, prog)
            data, lam, probes, reached, message, enc_s, prune, deblock, dec, dec_s, psnr, ssim = out
            png_b = jpeg = None
            if s.compare:
                prog.check()
                if on_progress:
                    on_progress("Comparing with PNG and JPEG...", 0.99)
                png_b = png_size(rgb)
                prog.check()
                jpeg = jpeg_at_psnr(rgb, psnr)
            return EncodeOutcome(data, lam, W, H, psnr, ssim, dec, probes, reached, message, time.time() - t_all, enc_s, dec_s,
                                 deblock, prune, s.mode, how, png_b, jpeg, original_file_bytes)
        except BaseException:
            self._drop_partitioner()
            raise

    def _prune_stats(self, part, pruned):
        if not pruned:
            return None
        st, nt = part.stats, max(1, part.stats["tiles"])
        nr = part.space.n_rects
        return dict(k=part.k, looked=100 * st["looked_at"] / nt / nr, priced=100 * st["priced_as_leaf"] / nt / nr,
                    ms=1000 * st["seconds_wall"] / nt)

    def _one_encode(self, rgb, prof, part, pruned, s, lam, prog, label):
        """One real encode at `lam`: (EncodeResult, seconds, prune stats)."""
        prog.start_probe(prog.probe, label)
        if pruned:
            part.reset_stats()
            part.progress = prog.phase("partitioning", 0.0, 0.7)
        t0 = time.time()
        res = encode_image(rgb, prof, lam, s.mu, progress=prog.phase("encoding", 0.7 if pruned else 0.0, 1.0), partitioner=part)
        return res, time.time() - t0, self._prune_stats(part, pruned)

    def _encode_policy(self, rgb, prof, part, pruned, s, prog):
        records = {}

        def encode_at(lam, label):
            res, sec, pr = self._one_encode(rgb, prof, part, pruned, s, lam, prog, label)
            dec = d_s = ps = None
            if s.mode == "psnr":
                t0 = time.time()
                dec = decode_image(res.data, prof)
                d_s = time.time() - t0
                ps = psnr_rgb(rgb, dec)
            records[lam] = (res, sec, pr, dec, ps, d_s)
            return res, ps

        reached, message = True, ""
        if s.mode == "quality":
            lam = s.lam
            encode_at(lam, f"λ {lam:.4f}")
            probes_order = [lam]
        else:
            counter = [0]

            def measure(lam):
                prog.probe = counter[0]
                counter[0] += 1
                res, ps = encode_at(lam, f"Step {counter[0]} · λ {lam:.4f}")
                return float(res.nbytes) if s.mode == "size" else ps

            target = float(s.target_bytes) if s.mode == "size" else float(s.target_psnr)
            probe_list, best, reached = solve_lambda(measure, s.mode, target)
            probes_order = [p[0] for p in probe_list]
            lam = probes_order[best]
        res, sec, pr, dec, ps, d_s = records[lam]
        if dec is None:
            t0 = time.time()
            dec = decode_image(res.data, prof)
            d_s = time.time() - t0
            ps = psnr_rgb(rgb, dec)
        ssim = ssim_y(rgb, dec)
        probes = [(l, records[l][0].nbytes, records[l][4]) for l in probes_order]
        if s.mode == "size":
            ratio = res.nbytes / s.target_bytes
            message = (f"{len(probes)} encodes to find λ; the file is {100 * abs(1 - ratio):.1f}% {'under' if ratio <= 1 else 'OVER'} the target"
                       if reached else _over_message(res.nbytes, s.target_bytes, len(probes)))
        elif s.mode == "psnr":
            message = (f"{len(probes)} encodes to find λ; PSNR is {ps - s.target_psnr:+.2f} dB from the target"
                       if reached else f"could not reach {s.target_psnr:.1f} dB: the best the search found is {ps:.2f} dB "
                                       f"(λ is limited to {LAM_MIN}..{LAM_MAX})")
        return res.data, lam, probes, reached, message, sec, pr, res.deblock, dec, d_s, ps, ssim

    def _encode_oracle_target(self, rgb, prof, s, prog):
        prog.start_probe(0, "Analysing")
        cb = prog.phase("analysing", 0.0, 1.0)
        t0 = time.time()
        if s.mode == "size":
            tr = encode_to_target(rgb, prof, max_bytes=int(s.target_bytes), mu_bits=s.mu, progress=cb)
        else:
            tr = encode_to_target(rgb, prof, target_psnr=float(s.target_psnr), mu_bits=s.mu, progress=cb)
        sec = time.time() - t0
        res = tr.result
        t0 = time.time()
        dec = decode_image(res.data, prof)
        d_s = time.time() - t0
        ps, ssim = psnr_rgb(rgb, dec), ssim_y(rgb, dec)
        reached = bool(tr.reached)
        if s.mode == "size":
            message = (f"file is {100 * (1 - res.nbytes / s.target_bytes):.1f}% under the target" if reached
                       else _over_message(res.nbytes, s.target_bytes, tr.iterations))
        else:
            message = (f"PSNR is {ps - s.target_psnr:+.2f} dB from the target" if reached
                       else f"could not reach {s.target_psnr:.1f} dB: best {ps:.2f} dB")
        return res.data, tr.lam, [(tr.lam, res.nbytes, ps)], reached, message, sec, None, res.deblock, dec, d_s, ps, ssim

    def decode(self, data: bytes, profile_dirs, on_progress=None) -> DecodeOutcome:
        try:
            info = read_info(data)
        except AmbiFormatError as e:
            raise EngineError(f"not a readable AMBI file: {e}") from None
        prof = self.profile_for(data, profile_dirs)
        if on_progress:
            on_progress("Decoding...", 0.0)
        t0 = time.time()
        try:
            rgb = decode_image(data, prof)
        except AmbiFormatError as e:
            raise EngineError(f"could not decode the file: {e}") from None
        info["bpp"] = info["total_bytes"] * 8 / (info["width"] * info["height"])
        return DecodeOutcome(rgb, info, prof.name, time.time() - t0)
