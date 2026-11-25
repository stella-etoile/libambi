from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ambi_core.bitstream import decode_image, encode_image, read_info
from ambi_core.color import psnr_rgb
from ambi_core.errors import AmbiFormatError
from ambi_core.metrics import ssim_y
from ambi_core.profile import (Profile, default_profile, find_profile, list_profiles,
                               profile_search_dirs)
from ambi_core.search import TileTables, encode_to_target

DEFAULT_LAMBDA = 0.03


def deblock_arg(text: str):
    if text == "auto":
        return None
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("expected 'auto' or an integer level 0-16") from None
    if not 0 <= v <= 16:
        raise argparse.ArgumentTypeError("deblock level must be 0-16")
    return v


def eprint(*a):
    print(*a, file=sys.stderr)


def make_progress(label: str, enabled: bool):
    if not enabled:
        return None
    t0 = time.time()

    def cb(done, total):
        sys.stderr.write(f"\r{label}: {done}/{total} tiles  ({time.time() - t0:.0f}s)")
        sys.stderr.flush()
        if done == total:
            sys.stderr.write("\n")
    return cb


def load_rgb(path: Path):
    im = Image.open(path)
    had_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
    return np.array(im.convert("RGB")), had_alpha


def pick_profile(spec, dirs):
    if spec:
        return find_profile(spec, dirs)
    profs = list_profiles(dirs)
    for p in profs:
        if p.name == "photo-demo":
            return p
    if profs:
        return profs[0]
    eprint("warning: no trained profile found; using the untrained default (compresses worse)")
    return default_profile()


def human(n: float) -> str:
    return f"{int(n):,}"


def cmd_encode(a) -> int:
    src = Path(a.input)
    rgb, had_alpha = load_rgb(src)
    H, W = rgb.shape[:2]
    prof = pick_profile(a.profile, a.profile_dir)
    out = Path(a.output) if a.output else src.with_suffix(".ambi")
    show = sys.stderr.isatty() and not a.quiet
    t0 = time.time()
    mode = "lambda"
    part = None
    if a.prune_k and not a.policy:
        raise ValueError("--prune-k needs --policy")
    if a.workers > 1 and not a.prune_k:
        raise ValueError("--workers needs --policy and --prune-k (it prices the pruned search's leaves in parallel)")
    if a.policy:
        if a.max_bytes is not None or a.target_psnr is not None:
            raise ValueError("--policy works with --lambda; the size/PSNR search needs the oracle's cost tables")
        try:
            from ambi_rl.pruned import make_partitioner
        except ImportError as e:
            raise ValueError(f"--policy needs PyTorch ({e})") from None
        part = make_partitioner(a.policy, prof, a.device, a.prune_k, a.workers)
        if a.prune_k:
            part.progress = make_progress("partitioning", show)
    if a.max_bytes is not None or a.target_psnr is not None:
        tr = encode_to_target(rgb, prof, max_bytes=a.max_bytes, target_psnr=a.target_psnr,
                              mu_bits=a.mu, deblock=a.deblock, progress=make_progress("analysing", show))
        res, lam = tr.result, tr.lam
        mode = "max-bytes" if a.max_bytes is not None else "target-psnr"
        if not tr.reached:
            eprint("warning: the target could not be met with this profile; "
                   "wrote the closest result")
    else:
        lam = a.lam if a.lam is not None else DEFAULT_LAMBDA
        res = encode_image(rgb, prof, lam, a.mu, progress=make_progress("encoding", show),
                           deblock=a.deblock, partitioner=part)
        if part is not None:
            mode = ("lambda, policy + exact search over its top %d actions" % a.prune_k) if a.prune_k else "lambda, learned partition policy"
    out.write_bytes(res.data)
    if hasattr(part, "close"):
        part.close()
    enc_s = time.time() - t0

    raw = H * W * 3
    on_disk = src.stat().st_size
    nbytes = res.nbytes
    info = {
        "input": str(src), "output": str(out), "width": W, "height": H,
        "profile": prof.name, "profile_id": prof.profile_id, "mode": mode, "lambda": lam,
        "input_file_bytes": on_disk, "raw_bytes": raw, "compressed_bytes": nbytes,
        "saved_vs_input_file_pct": 100 * (1 - nbytes / on_disk),
        "saved_vs_raw_pct": 100 * (1 - nbytes / raw),
        "ratio_vs_input_file": on_disk / nbytes, "ratio_vs_raw": raw / nbytes,
        "bpp": nbytes * 8 / (H * W), "encode_seconds": enc_s,
        "deblock_level": res.deblock, "deblock": "auto" if a.deblock is None else "forced",
    }
    if not a.no_verify:
        dec = decode_image(res.data, prof)
        info["psnr_db"] = psnr_rgb(rgb, dec)
        info["ssim_y"] = ssim_y(rgb, dec)
    if a.json:
        print(json.dumps(info, indent=2))
        return 0
    print(f"input:    {src}  ({W}x{H}, {human(on_disk)} bytes on disk, {human(raw)} bytes raw)")
    print(f"profile:  {prof.name} (id {prof.profile_id})   lambda {lam:.4g}   [{mode}]")
    print(f"output:   {out}  {human(nbytes)} bytes")
    print(f"saved:    {info['saved_vs_input_file_pct']:.1f}% vs input file ({info['ratio_vs_input_file']:.1f}:1)"
          f"   |   {info['saved_vs_raw_pct']:.1f}% vs raw ({info['ratio_vs_raw']:.1f}:1)")
    print(f"bitrate:  {info['bpp']:.3f} bpp")
    print(f"deblock:  level {res.deblock} ({info['deblock']})" + ("  (filter off)" if res.deblock == 0 else ""))
    if "psnr_db" in info:
        print(f"quality:  PSNR {info['psnr_db']:.2f} dB   SSIM(Y) {info['ssim_y']:.4f}   (measured by decoding the file)")
    if had_alpha:
        print("note:     alpha channel was dropped (AMBI stores RGB only)")
    print("note:     metadata (EXIF, colour profile) is not stored")
    return 0


def cmd_decode(a) -> int:
    src = Path(a.input)
    data = src.read_bytes()
    rgb = decode_image(data, profile_dirs=a.profile_dir, deblock=not a.no_deblock)
    out = Path(a.output) if a.output else src.with_suffix(".png")
    Image.fromarray(rgb).save(out)
    print(f"decoded {src} -> {out}  ({rgb.shape[1]}x{rgb.shape[0]})")
    return 0


def cmd_info(a) -> int:
    data = Path(a.input).read_bytes()
    info = read_info(data)
    try:
        p = find_profile(info["profile_id"], a.profile_dir)
        info["profile"] = {"name": p.name, "fingerprint_matches": f"{p.fingerprint:08x}" == info["fingerprint"]}
    except FileNotFoundError:
        info["profile"] = None
    info["bpp"] = info["total_bytes"] * 8 / (info["width"] * info["height"])
    if a.json:
        print(json.dumps(info, indent=2))
        return 0
    print(f"{a.input}: AMBI v{info['version']}, {info['width']}x{info['height']}, "
          f"{human(info['total_bytes'])} bytes ({info['bpp']:.3f} bpp)")
    print(f"  header {info['header_bytes']} bytes, payload {human(info['payload_bytes'])} bytes, "
          f"deblock level {info['deblock_level']}")
    pr = info["profile"]
    print(f"  profile id {info['profile_id']}, fingerprint {info['fingerprint']}: " +
          ("not found in profile directories" if pr is None else
           f"{pr['name']} ({'matches' if pr['fingerprint_matches'] else 'DOES NOT MATCH the file'})"))
    return 0


def cmd_profiles(a) -> int:
    dirs = [str(d) for d in profile_search_dirs(a.profile_dir)]
    print("searched:", ", ".join(dirs))
    profs = list_profiles(a.profile_dir)
    if not profs:
        print("no profiles found")
    for p in profs:
        c = p.cfg
        print(f"  id {p.profile_id:<3} {p.name:<20} tile {c.tile}, blocks {c.min_block}-{c.max_block}, "
              f"aspect <= {c.aspect_cap:g}:1, fingerprint {p.fingerprint:08x}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ambi", description="AMBI image codec")
    ap.add_argument("--profile-dir", action="append", default=[], help="extra directory of *.ambiprofile files")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="compress an image to .ambi")
    e.add_argument("input"); e.add_argument("-o", "--output")
    e.add_argument("-p", "--profile", help="profile name, id, or path (default: photo-demo if present)")
    g = e.add_mutually_exclusive_group()
    g.add_argument("--lambda", dest="lam", type=float, help=f"rate/quality tradeoff (default {DEFAULT_LAMBDA}; higher = smaller)")
    g.add_argument("--max-bytes", type=int, help="largest file size allowed; picks the best quality that fits")
    g.add_argument("--target-psnr", type=float, help="smallest file that reaches this PSNR (dB)")
    e.add_argument("--mu", type=float, default=0.0, help="extra per-block bit cost (bias toward larger blocks)")
    e.add_argument("--policy", metavar="FILE.ambipolicy",
                   help="use a trained partition policy instead of the exact search (fast; needs PyTorch; with --lambda only)")
    e.add_argument("--prune-k", type=int, default=0, metavar="K",
                   help="with --policy: run the exact search over the policy's K best actions per rectangle instead of trusting its "
                        "first choice (closer to the exact search, slower; 3 is a good start; 0 = off)")
    e.add_argument("--device", default="cpu", help="where the --policy network runs (cpu, cuda)")
    e.add_argument("--workers", type=int, default=1, metavar="N",
                   help="with --prune-k: N processes price the candidate leaves while the policy runs on --device (same result, much faster; "
                        "try the number of physical cores)")
    e.add_argument("--deblock", type=deblock_arg, default=None, metavar="{auto,0..16}",
                   help="deblocking filter level stored in the file (default auto: the encoder picks; 0 = off)")
    e.add_argument("--no-verify", action="store_true", help="skip decoding the result to measure quality")
    e.add_argument("--json", action="store_true"); e.add_argument("-q", "--quiet", action="store_true")
    e.set_defaults(fn=cmd_encode)

    d = sub.add_parser("decode", help="decompress .ambi to PNG")
    d.add_argument("input"); d.add_argument("-o", "--output")
    d.add_argument("--no-deblock", action="store_true", help="skip the deblocking filter (for A/B comparison)")
    d.set_defaults(fn=cmd_decode)

    i = sub.add_parser("info", help="show header information")
    i.add_argument("input"); i.add_argument("--json", action="store_true")
    i.set_defaults(fn=cmd_info)

    pr = sub.add_parser("profiles", help="list available profiles")
    pr.set_defaults(fn=cmd_profiles)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (AmbiFormatError, FileNotFoundError, OSError, ValueError) as e:
        eprint(f"ambi: error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
