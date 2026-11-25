from __future__ import annotations

import numpy as np
import torch

from ambi_core.config import CodecConfig
from ambi_core.rects import RectSpace
from .model import PolicyNet
from .rollout import rollout, trees_from_trace


def save_policy(model: PolicyNet, path, meta: dict | None = None) -> None:
    torch.save({"cfg": model.cfg_dict, "width": model.width, "hidden": model.hidden,
                "state": model.state_dict(), "meta": meta or {}}, path)


def load_policy(path, device="cpu") -> PolicyNet:
    ck = torch.load(path, map_location=device, weights_only=True)
    model = PolicyNet(CodecConfig.from_dict(ck["cfg"]), ck["width"], ck["hidden"])
    model.load_state_dict(ck["state"])
    return model.to(device).eval()


class PolicyPartitioner:
    def __init__(self, model: PolicyNet, device="cpu", batch: int = 64):
        self.model = model.to(device).eval()
        self.device, self.batch = device, batch
        self.space = RectSpace(CodecConfig.from_dict(model.cfg_dict))

    @classmethod
    def load(cls, path, device="cpu", batch: int = 64):
        return cls(load_policy(path, device), device, batch)

    @torch.no_grad()
    def trees(self, tiles: np.ndarray, lam: float):
        out = []
        for s in range(0, len(tiles), self.batch):
            x = torch.as_tensor(tiles[s:s + self.batch], dtype=torch.float32, device=self.device)
            lams = torch.full((x.shape[0],), float(lam), device=self.device)
            b, r, a = rollout(self.model, x, lams)
            out += trees_from_trace(self.space, b, r, a, x.shape[0])
        return out
