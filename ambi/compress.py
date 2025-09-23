# ambi/compress.py
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

# Per-record compression IDs (stored in bitstream)
COMP_NONE = 0
COMP_ZLIB = 1
# Reserve: 2=rangecoder, 3=zstd, 4=brotli, ...

class Compressor:
    """Interface for per-record compression."""
    comp_id: int = COMP_NONE
    name: str = "none"

    def banner(self) -> str:
        return f"{self.name}"

    def compress(self, raw: bytes) -> Tuple[int, bytes, int]:
        """Return (comp_id, payload_bytes, raw_len)."""
        return (COMP_NONE, raw, len(raw))

    def decompress(self, comp_id: int, payload: bytes, raw_len: Optional[int]) -> bytes:
        """Return raw bytes (length must equal raw_len when provided)."""
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
        # Only use if it helps (and avoids degenerate tiny overhead)
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
    """
    Returns a Compressor instance based on YAML or env.
      encoder.compression.type: zlib|none
      encoder.compression.level: 1..9 (zlib)
    Env override: AMBI_ZLIB_LEVEL
    """
    enc = cfg.get("encoder", {}) if cfg else {}
    comp = enc.get("compression", {}) if isinstance(enc.get("compression", {}), dict) else {}
    ctype = str(comp.get("type", "zlib")).lower()

    if ctype in ("none", "off", "0", "false"):
        return Compressor()

    if ctype in ("zlib", "deflate", "1", "true"):
        lvl = comp.get("level", None)
        if lvl is None:
            # env override wins
            lvl = int(os.environ.get("AMBI_ZLIB_LEVEL", "6"))
        return ZlibCompressor(level=int(lvl))

    # Fallback to none for unrecognized types (keeps decoding sane)
    return Compressor()