import struct
import numpy as np
from PIL import Image
from pathlib import Path
from ambi.utils import varint_encode, varint_decode

def load_image_rgb(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0
    return arr

def save_image_rgb(path: Path, arr: np.ndarray) -> None:
    x = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(x).save(path)

def rgb_to_ycbcr(x: np.ndarray) -> np.ndarray:
    r = x[..., 0]
    g = x[..., 1]
    b = x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return np.stack([y, cb, cr], axis=-1).astype(np.float32)

def ycbcr_to_rgb(x: np.ndarray) -> np.ndarray:
    y = x[..., 0]
    cb = x[..., 1] - 0.5
    cr = x[..., 2] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.stack([r, g, b], axis=-1).astype(np.float32)

class BitWriter:
    def __init__(self):
        self.buf = bytearray()
    def write_bytes(self, b: bytes):
        self.buf.extend(b)
    def write_u32(self, x: int):
        self.buf.extend(struct.pack(">I", x))
    def write_u16(self, x: int):
        self.buf.extend(struct.pack(">H", x))
    def write_u8(self, x: int):
        self.buf.extend(struct.pack(">B", x))
    def write_varint(self, x: int):
        self.buf.extend(varint_encode(x))
    def getvalue(self) -> bytes:
        return bytes(self.buf)

class BitReader:
    def __init__(self, data: bytes):
        self.data = memoryview(data)
        self.off = 0
    def read_bytes(self, n: int) -> bytes:
        b = self.data[self.off:self.off+n].tobytes()
        self.off += n
        return b
    def read_u32(self) -> int:
        x = struct.unpack(">I", self.read_bytes(4))[0]
        return x
    def read_u16(self) -> int:
        x = struct.unpack(">H", self.read_bytes(2))[0]
        return x
    def read_u8(self) -> int:
        x = struct.unpack(">B", self.read_bytes(1))[0]
        return x
    def read_varint(self) -> int:
        x, n = varint_decode(self.data[self.off:].tobytes())
        self.off += n
        return x

def fixed_quadtree_leaves(w: int, h: int, block: int):
    xs = list(range(0, w, block))
    ys = list(range(0, h, block))
    leaves = []
    for y in ys:
        for x in xs:
            bw = min(block, w - x)
            bh = min(block, h - y)
            leaves.append((x, y, bw, bh))
    return leaves

def write_header(bw: BitWriter, magic: str, version: int, w: int, h: int, block: int):
    bw.write_bytes(magic.encode("ascii"))
    bw.write_u16(version)
    bw.write_u32(w)
    bw.write_u32(h)
    bw.write_u16(block)

def read_header(br: BitReader):
    magic = br.read_bytes(4).decode("ascii")
    version = br.read_u16()
    w = br.read_u32()
    h = br.read_u32()
    block = br.read_u16()
    return magic, version, w, h, block