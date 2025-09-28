from __future__ import annotations
import struct
from typing import List, Tuple, Optional, Dict
from ambi.io import BitWriter, BitReader

FOOTER_MAGIC_V1 = b"IDX!"
FOOTER_MAGIC_V2 = b"IDX2"
FOOTER_PREFIX = b"AMBI"

__all__ = [
    "FOOTER_MAGIC_V1", "FOOTER_MAGIC_V2", "FOOTER_PREFIX",
    "write_index_footer_v1", "write_index_footer_v2", "read_index_footer"
]

def _build_table_bytes(chunk_offsets_counts: List[Tuple[int, int]]) -> bytes:
    tbl = BitWriter()
    tbl.write_bytes(FOOTER_PREFIX)
    tbl.write_varint(len(chunk_offsets_counts))
    for off, cnt in chunk_offsets_counts:
        tbl.write_varint(off)
        tbl.write_varint(cnt)
    return tbl.getvalue()

def write_index_footer_v1(bw: BitWriter, chunk_offsets_counts: List[Tuple[int, int]]) -> None:
    table_bytes = _build_table_bytes(chunk_offsets_counts)
    bw.write_bytes(table_bytes)
    bw.write_bytes(struct.pack("<Q", len(table_bytes)))
    bw.write_bytes(FOOTER_MAGIC_V1)

def write_index_footer_v2(bw: BitWriter, chunk_offsets_counts: List[Tuple[int, int]], payload_crc32: int) -> None:
    table_bytes = _build_table_bytes(chunk_offsets_counts)
    bw.write_bytes(table_bytes)
    bw.write_bytes(struct.pack("<I", int(payload_crc32) & 0xFFFFFFFF))
    bw.write_bytes(struct.pack("<Q", len(table_bytes)))
    bw.write_bytes(FOOTER_MAGIC_V2)

def _parse_table(table: bytes) -> List[Tuple[int, int]]:
    if not table.startswith(FOOTER_PREFIX):
        raise ValueError("Index table missing AMBI prefix")
    br = BitReader(table[len(FOOTER_PREFIX):])
    n_chunks = br.read_varint()
    pairs: List[Tuple[int, int]] = []
    for _ in range(n_chunks):
        off = br.read_varint()
        cnt = br.read_varint()
        pairs.append((off, cnt))
    return pairs

def read_index_footer(data: bytes) -> Optional[Dict]:
    if len(data) < 12:
        return None

    magic = data[-4:]
    if magic == FOOTER_MAGIC_V2:
        if len(data) < 4 + 8 + 4:
            return None
        table_size = struct.unpack("<Q", data[-12:-4])[0]
        crc32_val = struct.unpack("<I", data[-16:-12])[0]
        table_start = len(data) - 16 - table_size
        if table_start < 0:
            return None
        table = data[table_start: table_start + table_size]
        try:
            pairs = _parse_table(table)
        except Exception:
            return None
        ranges = []
        for i, (off, cnt) in enumerate(pairs):
            next_off = pairs[i+1][0] if i + 1 < len(pairs) else table_start
            ranges.append((off, next_off, cnt))
        return {
            "version": 2,
            "ranges": ranges,
            "table_start": table_start,
            "payload_crc32": crc32_val,
        }

    if magic == FOOTER_MAGIC_V1:
        table_size = struct.unpack("<Q", data[-12:-4])[0]
        table_start = len(data) - 12 - table_size
        if table_start < 0:
            return None
        table = data[table_start: table_start + table_size]
        try:
            pairs = _parse_table(table)
        except Exception:
            return None
        ranges = []
        for i, (off, cnt) in enumerate(pairs):
            next_off = pairs[i+1][0] if i + 1 < len(pairs) else table_start
            ranges.append((off, next_off, cnt))
        return {
            "version": 1,
            "ranges": ranges,
            "table_start": table_start,
            "payload_crc32": None,
        }

    return None