import numpy as np
import hashlib

def iter_blocks(img_ycc: np.ndarray, block: int):
    h, w, _ = img_ycc.shape
    xs = list(range(0, w, block))
    ys = list(range(0, h, block))
    for y in ys:
        for x in xs:
            bw = min(block, w - x)
            bh = min(block, h - y)
            yield x, y, img_ycc[y:y+bh, x:x+bw, :]

def quantize(x: np.ndarray, q: int) -> np.ndarray:
    s = max(1, int(q))
    return np.round(x * 255.0 / s).astype(np.int16)

def dequantize(qx: np.ndarray, q: int) -> np.ndarray:
    s = max(1, int(q))
    return (qx.astype(np.float32) * s) / 255.0

def hash_prefix(arr: np.ndarray, bits: int) -> int:
    if bits <= 0:
        return 0
    h = hashlib.sha256(arr.astype(np.float32).tobytes()).digest()
    val = int.from_bytes(h, "big")
    mask = (1 << bits) - 1
    shift = len(h) * 8 - bits
    return (val >> shift) & mask

class Budget:
    def __init__(self, bits: int):
        self.total = bits
        self.used = 0
    def take(self, n: int) -> bool:
        if self.used + n > self.total:
            return False
        self.used += n
        return True
    def left(self) -> int:
        return self.total - self.usedo