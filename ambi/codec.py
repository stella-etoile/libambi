# ambi/codec.py
from __future__ import annotations

import os
import math
import zlib
import numpy as np
from pathlib import Path
from collections import Counter, deque
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import shared_memory
from tqdm.auto import tqdm

from ambi.io import (
    load_image_rgb,
    save_image_rgb,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
    BitWriter,
    BitReader,
    write_header,
    read_header,
    fixed_quadtree_leaves,
)
from ambi.core import quantize, dequantize, hash_prefix
from ambi.features import block_features
from ambi.models import load_prior, load_policy, DeterministicPrior
from ambi.partition import dynamic_quadtree_leaves
from ambi.bitstream import (
    write_index_footer_v1, write_index_footer_v2, read_index_footer
)
from ambi.compress import (
    choose_compressor, Compressor, ZlibCompressor,
    COMP_NONE, COMP_ZLIB,
)

# -----------------------
# helpers
# -----------------------

def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2))

def _env_int(*keys, default=None):
    for k in keys:
        v = os.environ.get(k)
        if v and str(v).strip():
            try:
                return int(v)
            except Exception:
                pass
    return default

def _get_executor():
    try:
        import multiprocessing as mp
        if os.name == "posix":
            ctx = mp.get_context("fork")
            class ForkPool(ProcessPoolExecutor):
                def __init__(self, *a, **kw):
                    kw["mp_context"] = ctx
                    super().__init__(*a, **kw)
            return ForkPool
    except Exception:
        pass
    return ProcessPoolExecutor

# -----------------------
# ENCODE workers
# -----------------------

_ENC_SHM = None
_ENC_ARR = None
_ENC_POLICY = None
_ENC_PRIOR = None
_ENC_Q_DEFAULT = 12
_ENC_K_DEFAULT = 5
_ENC_H_DEFAULT = 8
_ENC_HAVE_COMP = False
_ENC_COMP = Compressor()  # polymorphic

def _enc_worker_init(shm_name: str, shape: tuple, dtype_str: str, cfg: dict):
    global _ENC_SHM, _ENC_ARR, _ENC_POLICY, _ENC_PRIOR
    global _ENC_Q_DEFAULT, _ENC_K_DEFAULT, _ENC_H_DEFAULT
    global _ENC_HAVE_COMP, _ENC_COMP

    _ENC_SHM = shared_memory.SharedMemory(name=shm_name)
    _ENC_ARR = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_ENC_SHM.buf)
    _ENC_POLICY = load_policy(cfg)
    _ENC_PRIOR = load_prior(cfg)
    enc = cfg.get("encoder", {}) if cfg else {}
    fmt = cfg.get("format", {}) if cfg else {}
    _ENC_Q_DEFAULT = int(enc.get("default_q", 12))
    _ENC_K_DEFAULT = int(enc.get("default_K", 5))
    _ENC_H_DEFAULT = int(enc.get("default_H", 8))
    _ENC_HAVE_COMP = int(fmt.get("version", 1)) >= 3
    _ENC_COMP = choose_compressor(cfg)

    if os.environ.get("AMBI_DEBUG"):
        print(f"[AMBI] worker start pid={os.getpid()} role=encode comp={_ENC_COMP.banner()} v3={_ENC_HAVE_COMP}")

def _encode_batch(batch):
    """
    batch = list[(i_leaf, (x, y, bw, bh)), ...]  ->  [(i_leaf, bytes)]
    Record layout (v3+):
      x,y,w,h,q,K,H,hp,refine_flag[,refine_idx], comp, [len_c, len_r, CBY] or [len_r, RBY]
    Legacy v1/v2 (no comp):
      x,y,w,h,q,K,H,hp,refine_flag[,refine_idx], len_r, RBY
    """
    ycc = _ENC_ARR
    out = []
    for i_leaf, (x, y, bwid, bhei) in batch:
        b = ycc[y:y+bhei, x:x+bwid, :]

        feats = block_features(b)
        act = _ENC_POLICY.act(feats)
        q = int(act.get("q", _ENC_Q_DEFAULT))
        K = int(act.get("K", _ENC_K_DEFAULT))

        qblk = quantize(b, q)
        coarse = dequantize(qblk, q)
        cands, scores = _ENC_PRIOR.propose(coarse, K=K)

        if hasattr(_ENC_POLICY, "decide_H"):
            H = int(_ENC_POLICY.decide_H(scores))
        else:
            H = int(act.get("H", _ENC_H_DEFAULT))

        if H == 0:
            hp = 0; refine_flag = 0; refine_idx = 0
        else:
            errs = [(_mse(c, b), j) for j, c in enumerate(cands)]
            errs.sort(key=lambda z: z[0])
            target_idx = int(errs[0][1])
            hp = int(hash_prefix(cands[target_idx], H))
            survivors = [j for j, c in enumerate(cands) if hash_prefix(c, H) == hp]
            if len(survivors) <= 1:
                refine_flag = 0; refine_idx = 0
            else:
                refine_flag = 1
                refine_idx = int(survivors.index(target_idx))

        by_raw = qblk.astype(np.int16).tobytes(order="C")

        comp_id, payload, raw_len = (_ENC_COMP.compress(by_raw) if _ENC_HAVE_COMP
                                     else (COMP_NONE, by_raw, len(by_raw)))

        bwb = BitWriter()
        bwb.write_varint(x); bwb.write_varint(y)
        bwb.write_varint(bwid); bwb.write_varint(bhei)
        bwb.write_varint(q); bwb.write_varint(K); bwb.write_varint(H)
        bwb.write_varint(hp)
        bwb.write_varint(refine_flag)
        if refine_flag:
            bwb.write_varint(refine_idx)

        if _ENC_HAVE_COMP:
            # v3+: write comp tag
            bwb.write_varint(comp_id)
            if comp_id == COMP_NONE:
                bwb.write_varint(raw_len)
                bwb.write_bytes(payload)
            elif comp_id == COMP_ZLIB:
                bwb.write_varint(len(payload))
                bwb.write_varint(raw_len)
                bwb.write_bytes(payload)
            else:
                # unknown comp -> store raw defensively
                bwb.write_varint(raw_len)
                bwb.write_bytes(by_raw)
        else:
            # legacy v1/v2 record: raw only
            bwb.write_varint(raw_len)
            bwb.write_bytes(by_raw)

        out.append((i_leaf, bwb.getvalue()))
    return out

# -----------------------
# DECODE workers
# -----------------------

_DEC_SHM = None
_DEC_REC = None
_DEC_IN_SHM = None
_DEC_IN_BUF = None
_DEC_HAVE_COMP = False
_DEC_COMP = Compressor()  # polymorphic

def _dec_worker_init(rec_shm_name: str, rec_shape: tuple, rec_dtype_str: str,
                     in_shm_name: str, in_size: int,
                     have_comp: bool, cfg: dict):
    global _DEC_SHM, _DEC_REC, _DEC_IN_SHM, _DEC_IN_BUF
    global _DEC_HAVE_COMP, _DEC_COMP
    _DEC_SHM = shared_memory.SharedMemory(name=rec_shm_name)
    _DEC_REC = np.ndarray(rec_shape, dtype=np.dtype(rec_dtype_str), buffer=_DEC_SHM.buf)
    _DEC_IN_SHM = shared_memory.SharedMemory(name=in_shm_name)
    _DEC_IN_BUF = np.ndarray((in_size,), dtype=np.uint8, buffer=_DEC_IN_SHM.buf)
    _DEC_HAVE_COMP = bool(have_comp)
    _DEC_COMP = choose_compressor(cfg)
    if os.environ.get("AMBI_DEBUG"):
        print(f"[AMBI] worker start pid={os.getpid()} role=decode comp={_DEC_COMP.banner()} v3={_DEC_HAVE_COMP}")

def _decode_records_bytes(records_bytes: bytes):
    """
    Parse a sequence of block records from a contiguous byte slice and write them
    into the shared output image.
    v3+: comp tag is present. v1/v2: raw length only.
    """
    buf = memoryview(records_bytes)
    n = len(buf); off = 0

    def read_uvarint(o: int) -> tuple[int, int]:
        val = 0; shift = 0
        while True:
            if o >= n:
                raise ValueError("truncated varint in chunk")
            b = buf[o]; o += 1
            val |= (int(b) & 0x7F) << shift
            if (b & 0x80) == 0:
                return val, o
            shift += 7
            if shift > 63:
                raise ValueError("varint too long")

    while off < n:
        x, off = read_uvarint(off)
        y, off = read_uvarint(off)
        bwid, off = read_uvarint(off)
        bhei, off = read_uvarint(off)
        q, off = read_uvarint(off)
        K, off = read_uvarint(off)
        H, off = read_uvarint(off)
        hp, off = read_uvarint(off)
        refine_flag, off = read_uvarint(off)
        if refine_flag:
            refine_idx, off = read_uvarint(off)
        else:
            refine_idx = 0

        # payload (with/without comp tag)
        if _DEC_HAVE_COMP:
            comp_id, off = read_uvarint(off)
            if comp_id == COMP_NONE:
                ln, off = read_uvarint(off)
                if off + ln > n:
                    raise ValueError("truncated raw payload")
                by = bytes(buf[off:off+ln]); off += ln
            elif comp_id == COMP_ZLIB:
                clen, off = read_uvarint(off)
                rlen, off = read_uvarint(off)
                if off + clen > n:
                    raise ValueError("truncated compressed payload")
                cbuf = bytes(buf[off:off+clen]); off += clen
                by = _DEC_COMP.decompress(comp_id, cbuf, rlen)
            else:
                raise ValueError(f"unknown comp_id={comp_id}")
        else:
            ln, off = read_uvarint(off)
            if off + ln > n:
                raise ValueError("truncated payload")
            by = bytes(buf[off:off+ln]); off += ln

        qblk = np.frombuffer(by, dtype=np.int16).reshape((bhei, bwid, 3))
        coarse = dequantize(qblk, q)
        prior = DeterministicPrior(K=K)
        cands, _ = prior.propose(coarse, K=K)

        if H == 0:
            idx = 0
        else:
            survivors = [j for j, c in enumerate(cands) if hash_prefix(c, H) == hp]
            if not survivors:
                idx = 0
            elif len(survivors) == 1:
                idx = survivors[0]
            else:
                idx = survivors[refine_idx] if 0 <= refine_idx < len(survivors) else survivors[0]

        _DEC_REC[y:y+bhei, x:x+bwid, :] = cands[idx]

def _decode_batch_from_index(args):
    start, end = args
    buf = _DEC_IN_BUF[start:end].tobytes()
    _decode_records_bytes(buf)
    return 1

# -----------------------
# public API
# -----------------------

def encode_image(inp: Path, outp: Path, cfg: dict):
    img = load_image_rgb(inp)
    ycc = rgb_to_ycbcr(img)

    enc = cfg.get("encoder", {}) if cfg else {}
    fmt = cfg.get("format", {}) if cfg else {}

    # tiling/quadtree
    block_default = int(enc.get("block_size", 32))
    min_block = int(enc.get("min_block", block_default))
    max_block = int(enc.get("max_block", block_default))
    use_quadtree = max_block > min_block

    magic = str(fmt.get("magic", "AMBI"))[:4]
    version = int(fmt.get("version", 1))

    ncpu = os.cpu_count() or 1
    num_workers = enc.get("num_workers", None)
    if isinstance(num_workers, str) and num_workers.lower() == "max":
        num_workers = ncpu
    if not isinstance(num_workers, int) or num_workers <= 0:
        num_workers = _env_int("AMBI_NUM_WORKERS", default=max(1, min(ncpu - 1, 16)))
    num_workers = max(1, min(int(num_workers), 64))

    # leaves
    if use_quadtree:
        split_cfg = enc.get("split", {}) if isinstance(enc.get("split", {}), dict) else {}
        var_th = float(split_cfg.get("var_thresh", os.environ.get("AMBI_SPLIT_VAR", 0.0025)))
        grad_th = float(split_cfg.get("grad_thresh", os.environ.get("AMBI_SPLIT_GRAD", 0.020)))
        leaves_xywh = dynamic_quadtree_leaves(ycc, min_block=min_block, max_block=max_block,
                                              var_thresh=var_th, grad_thresh=grad_th)
    else:
        h_img, w_img, _ = ycc.shape
        leaves_xywh = fixed_quadtree_leaves(w_img, h_img, block_default)

    sz_hist = Counter(f"{w}x{h}" for (_, _, w, h) in leaves_xywh)
    hsum = ", ".join(f"{k}:{v}" for k, v in sorted(sz_hist.items()))
    leaves = list(enumerate(leaves_xywh))
    n_blocks = len(leaves)

    user_chunk = enc.get("chunk_size", None)
    if user_chunk is None:
        chunk_size = max(32, min(2048, int(math.ceil(n_blocks / max(1, num_workers * 4)))))
    else:
        min_batches = max(2, num_workers * 2)
        max_chunk = max(32, int(math.floor(n_blocks / min_batches))) or 32
        chunk_size = min(int(user_chunk), max_chunk)

    nbatches = int(math.ceil(n_blocks / float(chunk_size)))
    mode = "qt" if use_quadtree else "fixed"

    # compressor (for banner only; workers init their own)
    have_comp = (version >= 3)
    comp = choose_compressor(cfg)
    comp_desc = comp.banner() if have_comp else "none(v<3)"

    print(f"[AMBI] encode workers={num_workers} chunk={chunk_size} blocks={n_blocks} "
          f"batches={nbatches} mode={mode} sizes{{{hsum}}} ver={version} comp={comp_desc}")

    # header
    header = BitWriter()
    h, w, _ = ycc.shape
    write_header(header, magic, version, w, h, min_block if use_quadtree else block_default)
    header.write_varint(n_blocks)
    header_bytes = header.getvalue()

    # share ycc to workers
    shm = shared_memory.SharedMemory(create=True, size=ycc.nbytes)
    try:
        np.ndarray(ycc.shape, dtype=ycc.dtype, buffer=shm.buf)[...] = ycc
        parts: list[bytes | None] = [None] * n_blocks

        if num_workers == 1:
            for start in range(0, n_blocks, chunk_size):
                batch = leaves[start:start+chunk_size]
                # run locally with a local copy of cfg (worker uses cfg for comp/version)
                global _ENC_HAVE_COMP, _ENC_COMP
                _ENC_HAVE_COMP = (version >= 3)
                _ENC_COMP = comp
                for i_leaf, blob in _encode_batch(batch):
                    parts[i_leaf] = blob
        else:
            initargs = (shm.name, ycc.shape, str(ycc.dtype), cfg)
            batches = [leaves[i:i+chunk_size] for i in range(0, n_blocks, chunk_size)]
            Pool = _get_executor()
            with Pool(max_workers=num_workers, initializer=_enc_worker_init, initargs=initargs) as ex:
                futs = [ex.submit(_encode_batch, b) for b in batches]
                for f in tqdm(as_completed(futs), total=len(futs),
                              desc=f"AMBI encode (w={num_workers}, c={chunk_size}, b={nbatches})",
                              unit="batch"):
                    for i_leaf, blob in f.result():
                        parts[i_leaf] = blob

        # payload + index
        payload_chunks = []
        offsets = []
        pos = len(header_bytes)
        for start in range(0, n_blocks, chunk_size):
            end = min(n_blocks, start + chunk_size)
            offsets.append((pos, end - start))
            for i in range(start, end):
                blob = parts[i]; assert blob is not None
                payload_chunks.append(blob)
                pos += len(blob)

        payload_bytes = b"".join(payload_chunks)
        payload_crc32 = zlib.crc32(payload_bytes) & 0xFFFFFFFF

        bw = BitWriter()
        bw.write_bytes(header_bytes)
        bw.write_bytes(payload_bytes)
        if version >= 2:
            write_index_footer_v2(bw, offsets, payload_crc32)
        else:
            write_index_footer_v1(bw, offsets)

        outp.write_bytes(bw.getvalue())

    finally:
        try: shm.close()
        except Exception: pass
        try: shm.unlink()
        except Exception: pass


def decode_image(inp: Path, outp: Path, num_workers: int | None = None, chunk_size: int | None = None):
    data = inp.read_bytes()
    br = BitReader(data)
    magic, version, w, h, block = read_header(br)
    n = br.read_varint()

    max_blocks = math.ceil(w / block) * math.ceil(h / block)
    if n <= 0 or n > max(1, max_blocks * 2):
        raise ValueError(f"Unreasonable block count n={n} (max ≈ {max_blocks})")

    ncpu = os.cpu_count() or 1
    if num_workers is None:
        envw = _env_int("AMBI_DECODE_WORKERS", "AMBI_NUM_WORKERS")
        num_workers = envw if envw else max(1, min(ncpu - 1, 16))
    num_workers = max(1, min(int(num_workers), 64))
    if chunk_size is None:
        chunk_size = max(32, min(2048, int(math.ceil(n / max(1, num_workers * 4)))))
    else:
        min_batches = max(2, num_workers * 2)
        max_chunk = max(32, int(math.floor(n / min_batches))) or 32
        chunk_size = min(int(chunk_size), max_chunk)

    have_comp = (version >= 3)
    comp = choose_compressor({"encoder": {"compression": {"type": "zlib"}}})  # default selection for display
    print(f"[AMBI] decode workers={num_workers} chunk={chunk_size} blocks={n} ver={version} comp={'enabled' if have_comp else 'none(v<3)'}")

    rec = np.zeros((h, w, 3), dtype=np.float32)
    rec_shm = shared_memory.SharedMemory(create=True, size=rec.nbytes)
    data_shm = shared_memory.SharedMemory(create=True, size=len(data))
    try:
        rec_view = np.ndarray(rec.shape, dtype=rec.dtype, buffer=rec_shm.buf)
        rec_view[...] = 0.0
        np.frombuffer(data_shm.buf, dtype=np.uint8)[:] = np.frombuffer(data, dtype=np.uint8)

        idx = read_index_footer(data)
        Pool = _get_executor()

        if idx and idx.get("ranges"):
            # CRC validate if available (v2)
            if idx.get("version") == 2 and idx.get("payload_crc32") is not None:
                ranges = idx["ranges"]
                table_start = int(idx["table_start"])
                first_start = min(s for (s, _e, _c) in ranges)
                computed = zlib.crc32(data[first_start:table_start]) & 0xFFFFFFFF
                stored = int(idx["payload_crc32"])
                if computed != stored:
                    raise ValueError(f"CRC mismatch: stored 0x{stored:08X} != computed 0x{computed:08X}")

            ranges = idx["ranges"]
            nbatches = len(ranges)
            with Pool(max_workers=num_workers, initializer=_dec_worker_init,
                      initargs=(rec_shm.name, rec.shape, str(rec.dtype),
                                data_shm.name, len(data), have_comp, {"encoder": {"compression": {"type": "zlib"}}})) as ex:
                futs = [ex.submit(_decode_batch_from_index, (s, e)) for (s, e, _cnt) in ranges]
                for _ in tqdm(as_completed(futs), total=len(futs),
                              desc=f"AMBI decode (indexed, w={num_workers}, b={nbatches})",
                              unit="chunk"):
                    pass
        else:
            # No footer: compatibility local path
            print("[AMBI] no index footer found; using compatibility path (slower)")
            nbatches = int(math.ceil(n / float(chunk_size)))
            pbar_read = tqdm(total=n,
                             desc=f"AMBI read (compat → decode w={num_workers}, c={chunk_size})",
                             unit="rec", leave=False)
            pbar_decode = tqdm(total=nbatches,
                               desc=f"AMBI decode (compat, w={num_workers}, b={nbatches})",
                               unit="batch")

            read_blocks = 0
            while read_blocks < n:
                take = min(chunk_size, n - read_blocks)
                chunk_writer = BitWriter()
                for _ in range(take):
                    x = br.read_varint(); y = br.read_varint()
                    bwid = br.read_varint(); bhei = br.read_varint()
                    q = br.read_varint(); K = br.read_varint(); H = br.read_varint()
                    hp = br.read_varint()
                    refine_flag = br.read_varint()
                    refine_idx = br.read_varint() if refine_flag else None
                    # legacy compat path assumes v<3 (raw only)
                    ln = br.read_varint()
                    by = br.read_bytes(ln)

                    # re-emit as v<3 raw record
                    chunk_writer.write_varint(x); chunk_writer.write_varint(y)
                    chunk_writer.write_varint(bwid); chunk_writer.write_varint(bhei)
                    chunk_writer.write_varint(q); chunk_writer.write_varint(K); chunk_writer.write_varint(H)
                    chunk_writer.write_varint(hp)
                    chunk_writer.write_varint(refine_flag)
                    if refine_idx is not None:
                        chunk_writer.write_varint(refine_idx)
                    chunk_writer.write_varint(ln)
                    chunk_writer.write_bytes(by)
                read_blocks += take
                pbar_read.update(take)

                _decode_records_bytes(chunk_writer.getvalue())
                pbar_decode.update(1)

            pbar_read.close()
            pbar_decode.close()

        rec[...] = rec_view

    finally:
        try: rec_shm.close()
        except Exception: pass
        try: rec_shm.unlink()
        except Exception: pass
        try: data_shm.close()
        except Exception: pass
        try: data_shm.unlink()
        except Exception: pass

    rgb = ycbcr_to_rgb(rec)
    save_image_rgb(outp, np.clip(rgb, 0.0, 1.0))