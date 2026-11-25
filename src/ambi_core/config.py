from __future__ import annotations

from dataclasses import dataclass, asdict
import math


INTRA_PRED_NAMES = ("dc", "v", "h", "tm", "dr", "drv", "drh")


@dataclass(frozen=True)
class CodecConfig:
    tile: int = 64
    grid: int = 4
    min_block: int = 4
    max_block: int = 64
    aspect_cap: float = 8.0

    qsteps: tuple = (4.0, 10.0, 24.0, 56.0)
    chroma_step_scale: float = 1.0
    rounding_offset: float = 0.4
    ts_max_area: int = 1024

    intra_preds: tuple = ()

    n_bands: int = 8
    prob_bits: int = 16
    ac_escape: int = 15
    dc_cats: int = 20

    alpha: float = 0.5
    mse_scale: float = 50.0
    ssim_win: int = 7

    deblock_alpha: float = 0.5
    deblock_beta: float = 0.1
    deblock_tc: float = 0.2
    deblock_flat_step: float = 10.0
    deblock_gain_step: float = 0.5
    deblock_max_level: int = 4

    def __post_init__(self):
        if self.min_block != self.grid:
            raise ValueError("v0 requires min_block == grid")
        if self.tile % self.grid or self.max_block % self.grid:
            raise ValueError("tile and max_block must be multiples of grid")
        if self.max_block > self.tile:
            raise ValueError("max_block must be <= tile")
        if self.ssim_win % 2 == 0:
            raise ValueError("ssim_win must be odd")
        if self.aspect_cap < 1:
            raise ValueError("aspect_cap must be >= 1")
        if len(set(self.intra_preds)) != len(self.intra_preds) or any(n not in INTRA_PRED_NAMES for n in self.intra_preds):
            raise ValueError(f"intra_preds must be distinct names from {INTRA_PRED_NAMES}, got {self.intra_preds}")
        if self.deblock_max_level < 0 or self.deblock_max_level > 16:
            raise ValueError("deblock_max_level must be in 0..16")
        if self.deblock_max_level > 0 and self.min_block < 4:
            raise ValueError("the deblocking filter needs min_block >= 4")

    @property
    def n_cells(self) -> int:
        return self.tile // self.grid

    @property
    def max_cells(self) -> int:
        return self.max_block // self.grid

    @property
    def n_q(self) -> int:
        return len(self.qsteps)

    @property
    def total(self) -> int:
        return 1 << self.prob_bits

    @property
    def n_intra(self) -> int:
        return len(self.intra_preds)

    @property
    def intra_base(self) -> int:
        return 2 * self.n_q + 1

    @property
    def n_modes(self) -> int:
        return 2 * self.n_q + 1 + self.n_intra * self.n_q

    def mode_name(self, m: int) -> str:
        if m < self.n_q:
            return f"dct{m}"
        if m == self.n_q:
            return "flat"
        if m < self.intra_base:
            return f"ts{m - self.n_q - 1}"
        return f"intra_{self.intra_preds[self.mode_pred(m)]}{self.mode_q(m)}"

    def mode_family(self, m: int) -> str:
        if m < self.n_q:
            return "dct"
        if m == self.n_q:
            return "flat"
        if m < self.intra_base:
            return "ts"
        return "intra"

    def mode_q(self, m: int) -> int:
        if m < self.n_q:
            return m
        if m == self.n_q:
            return -1
        if m < self.intra_base:
            return m - self.n_q - 1
        return (m - self.intra_base) % self.n_q

    def mode_pred(self, m: int) -> int:
        """Predictor index (into intra_preds) of an intra mode."""
        return (m - self.intra_base) // self.n_q

    # ---- shape rules (cells) ----
    def leaf_legal(self, w: int, h: int) -> bool:
        """Can a w x h (cells) rectangle be a leaf?"""
        if w > self.max_cells or h > self.max_cells:
            return False
        return max(w, h) <= self.aspect_cap * min(w, h)

    def legal_modes(self, w: int, h: int) -> list[int]:
        modes = list(range(self.n_q)) + [self.n_q]
        if (w * h) * self.grid * self.grid <= self.ts_max_area:
            modes += list(range(self.n_q + 1, 2 * self.n_q + 1))
        modes += list(range(self.intra_base, self.n_modes))
        return modes

    def to_dict(self) -> dict:
        d = asdict(self)
        d["qsteps"] = list(d["qsteps"])
        d["intra_preds"] = list(d["intra_preds"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CodecConfig":
        d = dict(d)
        d["qsteps"] = tuple(d["qsteps"])
        d["intra_preds"] = tuple(d.get("intra_preds", ()))
        return cls(**d)

    DECODER_FIELDS = ("tile", "grid", "min_block", "max_block", "aspect_cap", "qsteps",
                      "chroma_step_scale", "ts_max_area", "n_bands", "prob_bits",
                      "ac_escape", "dc_cats", "deblock_alpha", "deblock_beta", "deblock_tc",
                      "deblock_flat_step", "deblock_gain_step")

    def decoder_dict(self) -> dict:
        d = self.to_dict()
        out = {k: d[k] for k in self.DECODER_FIELDS}
        if self.intra_preds:
            out["intra_preds"] = list(self.intra_preds)
        return out


def bits_for_choices(n: int) -> float:
    """Uniform-code cost for choosing one of n options."""
    return math.log2(n) if n > 1 else 0.0
