from __future__ import annotations

import math

_TOP = 1 << 24
_MASK32 = 0xFFFFFFFF


class RangeEncoder:
    def __init__(self):
        self.low = 0
        self.range = _MASK32
        self.cache = 0
        self.cache_size = 1
        self.out = bytearray()
        self.ideal_bits = 0.0

    def _shift_low(self):
        if self.low < 0xFF000000 or self.low >= (1 << 32):
            carry = self.low >> 32
            temp = self.cache
            while True:
                self.out.append((temp + carry) & 0xFF)
                temp = 0xFF
                self.cache_size -= 1
                if self.cache_size == 0:
                    break
            self.cache = (self.low >> 24) & 0xFF
        self.cache_size += 1
        self.low = (self.low & 0x00FFFFFF) << 8

    def encode(self, start: int, freq: int, total_bits: int) -> None:
        r = self.range >> total_bits
        self.low += start * r
        self.range = r * freq
        self.ideal_bits += total_bits - math.log2(freq)
        while self.range < _TOP:
            self.range = (self.range << 8) & _MASK32
            self._shift_low()

    def encode_symbol(self, cdf, s: int, total_bits: int) -> None:
        self.encode(int(cdf[s]), int(cdf[s + 1] - cdf[s]), total_bits)

    def encode_raw(self, value: int, nbits: int) -> None:
        """Equiprobable bits (chunks of <=16)."""
        while nbits > 0:
            n = min(nbits, 16)
            nbits -= n
            self.encode((value >> nbits) & ((1 << n) - 1), 1, n)

    def encode_uniform(self, k: int, n: int) -> None:
        if n <= 1:
            return
        r = self.range // n
        self.low += k * r
        self.range = r
        self.ideal_bits += math.log2(n)
        while self.range < _TOP:
            self.range = (self.range << 8) & _MASK32
            self._shift_low()

    def finish(self) -> bytes:
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class RangeDecoder:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.range = _MASK32
        self.code = 0
        for _ in range(5):
            self.code = ((self.code << 8) | self._next()) & _MASK32

    def _next(self) -> int:
        b = self.data[self.pos] if self.pos < len(self.data) else 0
        self.pos += 1
        return b

    def decode_symbol(self, cdf, total_bits: int) -> int:
        r = self.range >> total_bits
        v = min(self.code // r, (1 << total_bits) - 1)
        lo, hi = 0, len(cdf) - 2
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if cdf[mid] <= v:
                lo = mid
            else:
                hi = mid - 1
        s = lo
        self.code -= int(cdf[s]) * r
        self.range = r * int(cdf[s + 1] - cdf[s])
        self._normalize()
        return s

    def decode_uniform(self, n: int) -> int:
        if n <= 1:
            return 0
        r = self.range // n
        v = min(self.code // r, n - 1)
        self.code -= v * r
        self.range = r
        self._normalize()
        return v

    def decode_raw(self, nbits: int) -> int:
        out = 0
        while nbits > 0:
            n = min(nbits, 16)
            nbits -= n
            r = self.range >> n
            v = min(self.code // r, (1 << n) - 1)
            self.code -= v * r
            self.range = r
            self._normalize()
            out = (out << n) | v
        return out

    def _normalize(self):
        while self.range < _TOP:
            self.range = (self.range << 8) & _MASK32
            self.code = ((self.code << 8) | self._next()) & _MASK32
