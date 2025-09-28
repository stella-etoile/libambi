from __future__ import annotations
import os
import zlib
from typing import Tuple, Optional, Dict, Any

__all__ = [
    "Compressor",
    "ZlibCompressor",
    "choose_compressor",
    "COMP_NONE",
    "COMP_ZLIB",
]

COMP_NONE = 0
COMP_ZLIB = 1

class Compressor:
    comp_id: int = COMP_NONE
    name: str = "none"

    def banner(self) -> str:
        return f"{self.name}"

    def compress(self, raw: bytes) -> Tuple[int, bytes, int]:
        return (COMP_NONE, raw, len(raw))

    def decompress(self, comp_id: int, payload: bytes, raw_len: Optional[int]) -> bytes:
        return payload

class ZlibCompressor(Compressor):
    comp_id = COMP_ZLIB
    name = "zlib"

    def __init__(self, level: int = 6):
        self.level = int(level)

    def banner(self) -> str:
        return f"{self.name}(level={self.level})"

    def compress(self, raw: bytes) -> Tuple[int, bytes, int]:
        c = zlib.compress(raw, self.level)
        if len(c) + 3 < len(raw):
            return (self.comp_id, c, len(raw))
        else:
            return (COMP_NONE, raw, len(raw))

    def decompress(self, comp_id: int, payload: bytes, raw_len: Optional[int]) -> bytes:
        if comp_id == COMP_NONE:
            return payload
        if comp_id != self.comp_id:
            raise ValueError(f"ZlibCompressor got unexpected comp_id={comp_id}")
        out = zlib.decompress(payload)
        if raw_len is not None and len(out) != int(raw_len):
            raise ValueError("zlib: decompressed length mismatch")
        return out


def choose_compressor(cfg: Optional[Dict[str, Any]]) -> Compressor:
    enc = cfg.get("encoder", {}) if cfg else {}
    comp = enc.get("compression", {}) if isinstance(enc.get("compression", {}), dict) else {}
    ctype = str(comp.get("type", "zlib")).lower()

    if ctype in ("none", "off", "0", "false"):
        return Compressor()

    if ctype in ("zlib", "deflate", "1", "true"):
        lvl = comp.get("level", None)
        if lvl is None:
            lvl = int(os.environ.get("AMBI_ZLIB_LEVEL", "6"))
        return ZlibCompressor(level=int(lvl))

    return Compressor()