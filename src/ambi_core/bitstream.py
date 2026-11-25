from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .blockio import decode_block, encode_block
from .color import rgb_to_ycc, ycc_to_rgb
from .deblock import choose_level, deblock as apply_deblock, rect_steps
from .entropy import area_bucket
from .errors import AmbiFormatError
from .intra import gather_refs, new_ext, predict
from .modes import LeafCoder
from .oracle import (LeafEvaluator, Node, encode_tile_closed_loop, pad_luma, reconstruct_tile,
                     result_from_tree, solve_table, solve_tile)
from .profile import Profile, find_profile
from .rangecoder import RangeDecoder, RangeEncoder

MAGIC = b"AMBI"
VERSION = 1
MAX_PIXELS = 1 << 28
MAX_DEBLOCK_LEVEL = 16


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _read_varint(data: bytes, pos: int):
    n = shift = 0
    while True:
        if pos >= len(data) or shift > 35:
            raise AmbiFormatError("truncated or corrupt header")
        b = data[pos]
        pos += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            return n, pos
        shift += 7


@dataclass
class Header:
    version: int
    profile_id: int
    fingerprint: int
    width: int
    height: int
    colorspace: int = 0
    deblock: int = 0


def pack_header(h: Header) -> bytes:
    return (MAGIC + bytes([h.version, h.profile_id]) + struct.pack("<I", h.fingerprint)
            + _varint(h.width) + _varint(h.height) + bytes([h.colorspace, h.deblock]))


def parse_header(data: bytes):
    if len(data) < 12 or data[:4] != MAGIC:
        raise AmbiFormatError("not an AMBI file (bad magic)")
    version, pid = data[4], data[5]
    if version != VERSION:
        raise AmbiFormatError(f"unsupported AMBI format version {version} "
                              f"(this build reads version {VERSION}; version 0 was a pre-release layout)")
    (fp,) = struct.unpack("<I", data[6:10])
    w, pos = _read_varint(data, 10)
    h, pos = _read_varint(data, pos)
    if pos + 1 >= len(data):
        raise AmbiFormatError("truncated header")
    cs, db = data[pos], data[pos + 1]
    pos += 2
    if w < 1 or h < 1 or w * h > MAX_PIXELS:
        raise AmbiFormatError(f"implausible image size {w}x{h}")
    if cs != 0:
        raise AmbiFormatError(f"unsupported colour space {cs}")
    if db > MAX_DEBLOCK_LEVEL:
        raise AmbiFormatError(f"corrupt header (deblock level {db})")
    return Header(version, pid, fp, w, h, cs, db), pos


def encode_tree(enc: RangeEncoder, cfg, model, coder: LeafCoder, tile: np.ndarray, node: Node) -> None:
    g = cfg.grid
    w, h = node.w, node.h
    legal = cfg.leaf_legal(w, h)
    can_cut = w > 1 or h > 1
    if legal and can_cut:
        enc.encode_uniform(1 if node.kind == "leaf" else 0, 2)
    if node.kind == "leaf":
        modes = cfg.legal_modes(w, h)
        if cfg.intra_preds:
            enc.encode_symbol(model.cdf("mode", (area_bucket(h * g, w * g),)), node.mode, cfg.prob_bits)
            qv = None if node.qv is None else node.qv[None]
            mean = None if node.mean is None else node.mean[None]
        else:
            enc.encode_uniform(modes.index(node.mode), len(modes))
            blk = tile[None, :, node.y * g:(node.y + h) * g, node.x * g:(node.x + w) * g]
            qv, mean = coder.quantize(node.mode, blk)
        encode_block(enc, cfg, model, node.mode, h * g, w * g,
                     None if qv is None else qv[0], None if mean is None else mean[0])
        return
    if w > 1 and h > 1:
        enc.encode_uniform(0 if node.kind == "v" else 1, 2)
    enc.encode_uniform(node.k - 1, (w - 1) if node.kind == "v" else (h - 1))
    encode_tree(enc, cfg, model, coder, tile, node.a)
    encode_tree(enc, cfg, model, coder, tile, node.b)


def decode_tree(dec: RangeDecoder, cfg, model, x, y, w, h, out: list) -> None:
    g = cfg.grid
    legal = cfg.leaf_legal(w, h)
    can_cut = w > 1 or h > 1
    is_leaf = True
    if legal and can_cut:
        is_leaf = bool(dec.decode_uniform(2))
    elif not legal:
        is_leaf = False
        if not can_cut:
            raise AmbiFormatError("corrupt data (illegal 1x1 leaf)")
    if is_leaf:
        modes = cfg.legal_modes(w, h)
        if cfg.intra_preds:
            mode = dec.decode_symbol(model.cdf("mode", (area_bucket(h * g, w * g),)), cfg.prob_bits)
            if mode not in modes:
                raise AmbiFormatError("corrupt data (illegal leaf mode)")
        else:
            mode = modes[dec.decode_uniform(len(modes))]
        qv, mean = decode_block(dec, cfg, model, mode, h * g, w * g)
        out.append((x * g, y * g, w * g, h * g, mode, qv, mean))
        return
    axis_v = True
    if w > 1 and h > 1:
        axis_v = dec.decode_uniform(2) == 0
    elif h == 1:
        axis_v = True
    else:
        axis_v = False
    k = dec.decode_uniform((w - 1) if axis_v else (h - 1)) + 1
    if axis_v:
        decode_tree(dec, cfg, model, x, y, k, h, out)
        decode_tree(dec, cfg, model, x + k, y, w - k, h, out)
    else:
        decode_tree(dec, cfg, model, x, y, w, k, out)
        decode_tree(dec, cfg, model, x, y + k, w, h - k, out)


@dataclass
class EncodeResult:
    data: bytes
    est_bits: float
    n_tiles: int
    lam: float
    mu_bits: float
    deblock: int = 0
    deblock_scores: tuple = ()

    @property
    def nbytes(self) -> int:
        return len(self.data)


@dataclass
class ImagePlan:
    """Everything the encoder decides before any bits are written."""
    ycc: np.ndarray
    recon: np.ndarray
    rects: list
    results: list
    height: int
    width: int
    ty: int
    tx: int

    @property
    def est_bits(self) -> float:
        return float(sum(r.total_bits for r in self.results))


def pad_to_tiles(ycc: np.ndarray, tile: int) -> np.ndarray:
    _, h, w = ycc.shape
    ph, pw = (-h) % tile, (-w) % tile
    if ph or pw:
        ycc = np.pad(ycc, ((0, 0), (0, ph), (0, pw)), mode="edge")
    return ycc


def tile_grid(height: int, width: int, tile: int):
    return -(-height // tile), -(-width // tile)


def plan_image(rgb: np.ndarray, profile: Profile, lam: float, mu_bits: float = 0.0,
               tables: list | None = None, progress=None, partitioner=None) -> ImagePlan:
    """Choose every tile's partition and modes and reconstruct the result (no bitstream yet).
    partitioner: optional object with trees(tiles (N,3,T,T) float32 YCbCr, lam) -> list[Node]
    (a learned policy). It decides the partitions; leaf modes are then chosen exactly as the oracle
    would. Without it, the exact oracle decides."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("encode_image expects an (H, W, 3) uint8 RGB array")
    cfg, model = profile.cfg, profile.model
    H, W = rgb.shape[:2]
    ycc = pad_to_tiles(rgb_to_ycc(rgb), cfg.tile)
    ty, tx = tile_grid(H, W, cfg.tile)
    T = cfg.tile
    recon = np.zeros_like(ycc)
    rects, results = [], []
    trees = None
    if partitioner is not None:
        if tables is not None:
            raise ValueError("give either cost tables (oracle) or a partitioner, not both")
        all_tiles = np.stack([ycc[:, j * T:(j + 1) * T, i * T:(i + 1) * T] for j in range(ty) for i in range(tx)])
        trees = partitioner.trees(np.ascontiguousarray(all_tiles), lam)
        ev = LeafEvaluator(cfg, model)
    intra = bool(cfg.intra_preds)
    if intra:
        ext_orig = new_ext(ycc.shape[1], ycc.shape[2])
        ext_orig[:, 1:, 1:] = ycc
        ext_rec = new_ext(ycc.shape[1], ycc.shape[2])
        coder64 = LeafCoder(cfg, model, dtype=np.float64)
    for j in range(ty):
        for i in range(tx):
            idx = j * tx + i
            tile = np.ascontiguousarray(ycc[:, j * T:(j + 1) * T, i * T:(i + 1) * T])
            if intra:
                if trees is not None:
                    root = trees[idx]
                elif tables is not None:
                    root = solve_table(cfg, tables[idx], lam, mu_bits).tree
                else:
                    root = solve_tile(cfg, model, tile, lam, mu_bits,
                                      ext=ext_orig[:, j * T:(j + 1) * T + 1, i * T:(i + 1) * T + 1]).tree
                res = encode_tile_closed_loop(cfg, model, tile, pad_luma(cfg, tile), ext_rec, i * T, j * T,
                                              root, lam, mu_bits, coder64)
                rc = ext_rec[:, 1 + j * T:1 + (j + 1) * T, 1 + i * T:1 + (i + 1) * T]
            else:
                if trees is not None:
                    res = result_from_tree(cfg, model, tile, trees[idx], lam, mu_bits, ev)
                else:
                    res = (solve_table(cfg, tables[idx], lam, mu_bits) if tables is not None
                           else solve_tile(cfg, model, tile, lam, mu_bits))
                rc, _ = reconstruct_tile(cfg, model, tile, res.leaves)
            recon[:, j * T:(j + 1) * T, i * T:(i + 1) * T] = rc
            rects += [(i * T + l.x, j * T + l.y, l.w, l.h, l.mode) for l in res.leaves]
            results.append(res)
            if progress:
                progress(idx + 1, ty * tx)
    return ImagePlan(ycc, recon, rects, results, H, W, ty, tx)


def encode_image(rgb: np.ndarray, profile: Profile, lam: float, mu_bits: float = 0.0,
                 tables: list | None = None, progress=None, deblock: int | None = None,
                 partitioner=None) -> EncodeResult:
    """rgb: (H, W, 3) uint8. `tables`: optional per-tile cost tables (raster order) to reuse.
    deblock: None = encoder picks the level that minimises the image distortion; an int forces it."""
    cfg, model = profile.cfg, profile.model
    plan = plan_image(rgb, profile, lam, mu_bits, tables, progress, partitioner)
    scores = ()
    if deblock is None:
        if cfg.deblock_max_level > 0:
            deblock, sc = choose_level(cfg, plan.ycc, plan.recon, rect_steps(cfg, plan.rects),
                                       plan.height, plan.width)
            scores = tuple(sc)
        else:
            deblock = 0
    elif not 0 <= deblock <= MAX_DEBLOCK_LEVEL:
        raise ValueError(f"deblock level must be in 0..{MAX_DEBLOCK_LEVEL}")
    coder = LeafCoder(cfg, model)
    enc = RangeEncoder()
    T = cfg.tile
    for j in range(plan.ty):
        for i in range(plan.tx):
            res = plan.results[j * plan.tx + i]
            tile = np.ascontiguousarray(plan.ycc[:, j * T:(j + 1) * T, i * T:(i + 1) * T])
            encode_tree(enc, cfg, model, coder, tile, res.tree)
    payload = enc.finish()
    head = pack_header(Header(VERSION, profile.profile_id, profile.fingerprint,
                              plan.width, plan.height, 0, deblock))
    return EncodeResult(head + payload, plan.est_bits, plan.ty * plan.tx, lam, mu_bits,
                        deblock, scores)


def read_info(data: bytes) -> dict:
    h, pos = parse_header(data)
    return {"version": h.version, "profile_id": h.profile_id,
            "fingerprint": f"{h.fingerprint:08x}", "width": h.width, "height": h.height,
            "deblock_level": h.deblock, "header_bytes": pos, "payload_bytes": len(data) - pos,
            "total_bytes": len(data)}


def decode_image(data: bytes, profile=None, profile_dirs=(), deblock: bool = True) -> np.ndarray:
    """-> (H, W, 3) uint8. `profile` may be a Profile; otherwise it is looked up by the header's id.
    deblock=False skips the post filter (for A/B comparison; the normal output has it applied)."""
    head, pos = parse_header(data)
    if profile is None or not isinstance(profile, Profile):
        try:
            profile = find_profile(head.profile_id, profile_dirs)
        except FileNotFoundError as e:
            raise AmbiFormatError(f"{e}") from None
    if profile.fingerprint != head.fingerprint:
        raise AmbiFormatError(
            f"profile mismatch: file was encoded with fingerprint {head.fingerprint:08x}, "
            f"profile {profile.name!r} has {profile.fingerprint:08x}")
    cfg, model = profile.cfg, profile.model
    T = cfg.tile
    ty, tx = tile_grid(head.height, head.width, T)
    payload = data[pos:]
    dec = RangeDecoder(payload)
    coder = LeafCoder(cfg, model, dtype=np.float64)
    ext = new_ext(ty * T, tx * T)
    rects = []
    for j in range(ty):
        for i in range(tx):
            leaves: list = []
            decode_tree(dec, cfg, model, 0, 0, cfg.n_cells, cfg.n_cells, leaves)
            for (x, y, w, h, mode, qv, mean) in leaves:
                fam, qi = cfg.mode_family(mode), cfg.mode_q(mode)
                X, Y = i * T + x, j * T + y
                if fam == "dct":
                    rc = coder.recon_dct(qv[None], qi)
                elif fam == "flat":
                    rc = coder.recon_flat(mean[None], h, w)
                elif fam == "intra":
                    above, left, corner = gather_refs(ext, [Y], [X], h, w)
                    pred = predict(cfg.intra_preds[cfg.mode_pred(mode)], above, left, corner, h, w)
                    rc = coder.recon_intra(qv[None], qi, pred)
                else:
                    rc = coder.recon_ts(qv[None], mean[None], qi)
                ext[:, 1 + Y:1 + Y + h, 1 + X:1 + X + w] = rc[0]
                rects.append((X, Y, w, h, mode))
    canvas = ext[:, 1:, 1:].copy()
    if dec.pos > len(payload) + 1:
        raise AmbiFormatError("file is truncated")
    if deblock and head.deblock > 0:
        canvas = apply_deblock(canvas, rect_steps(cfg, rects), cfg, head.deblock)
    return ycc_to_rgb(canvas[:, :head.height, :head.width])
