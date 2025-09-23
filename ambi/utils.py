import random
import time
import numpy as np

def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    except Exception:
        pass

def now_s():
    return time.perf_counter()

def varint_encode(value: int) -> bytes:
    x = int(value)
    out = bytearray()
    while True:
        to_write = x & 0x7F
        x >>= 7
        if x:
            out.append(to_write | 0x80)
        else:
            out.append(to_write)
            break
    return bytes(out)

def varint_decode(buf: bytes):
    shift = 0
    result = 0
    for i, b in enumerate(buf):
        result |= ((b & 0x7F) << shift)
        if not (b & 0x80):
            return result, i + 1
        shift += 7
    return result, len(buf)