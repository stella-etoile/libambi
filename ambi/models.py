# ambi/models.py
from __future__ import annotations
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

# -----------------------------
# Prior: small deterministic candidate bank
# -----------------------------

def _blur5(x: np.ndarray) -> np.ndarray:
    x0 = x.astype(np.float32, copy=False)
    # 4-neighborhood average + self
    x1 = (np.roll(x0, 1, 0) + np.roll(x0, -1, 0) +
          np.roll(x0, 1, 1) + np.roll(x0, -1, 1) + x0) / 5.0
    return x1

def _sharpen(x: np.ndarray, a: float = 0.5) -> np.ndarray:
    b = _blur5(x)
    return np.clip(x + a * (x - b), 0.0, 1.0)

def _contrast(x: np.ndarray, s: float = 1.08) -> np.ndarray:
    m = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - m) * float(s) + m, 0.0, 1.0)

def _contrast_down(x: np.ndarray, s: float = 0.92) -> np.ndarray:
    m = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - m) * float(s) + m, 0.0, 1.0)

def _edge_boost(x: np.ndarray, a: float = 0.35) -> np.ndarray:
    b = _blur5(x)
    e = x - b
    return np.clip(x + a * e, 0.0, 1.0)

def _cand_bank(x: np.ndarray) -> List[np.ndarray]:
    # Deterministic bank; first item is identity (top-1 prior)
    return [
        x,
        _blur5(x),
        _sharpen(x),
        _contrast(x, 1.08),
        _contrast_down(x, 0.92),
        _edge_boost(x, 0.35),
    ]

class Prior:
    """Abstract prior interface."""
    def __init__(self, K: int = 1):
        self.K = int(K)

    def propose(self, coarse_block: np.ndarray, K: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray]:
        raise NotImplementedError

class DeterministicPrior(Prior):
    """
    Simple prior that returns first-K transforms from a small bank,
    ordered by negative MSE to the coarse block (best first).
    """
    def propose(self, coarse_block: np.ndarray, K: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray]:
        k = int(K if K is not None else self.K)
        bank = _cand_bank(coarse_block)
        arr = bank[:max(1, min(k, len(bank)))]
        scores = []
        base = coarse_block.astype(np.float32, copy=False)
        for c in arr:
            d = float(np.mean((c.astype(np.float32, copy=False) - base) ** 2))
            scores.append(-d)  # higher is better
        order = np.argsort(-np.asarray(scores))
        arr2 = [arr[i] for i in order]
        scores2 = np.asarray([scores[i] for i in order], dtype=np.float32)
        return arr2, scores2

def load_prior(cfg: Dict[str, Any] | None) -> Prior:
    pc = cfg.get("prior", {}) if cfg else {}
    t = str(pc.get("type", "deterministic")).lower()
    K = int(pc.get("K", 5))
    # (Only deterministic provided in stub)
    return DeterministicPrior(K=K)

# -----------------------------
# Policies
# -----------------------------

class Policy:
    """Base policy: returns dict with q, K, H."""
    def __init__(self, q: int = 12, K: int = 5, H: int = 8):
        self.q = int(q); self.K = int(K); self.H = int(H)

    def act(self, features: np.ndarray) -> Dict[str, int]:
        return {"q": self.q, "K": self.K, "H": self.H}

class FixedMarginPolicy(Policy):
    """
    H decision based on score margin from the prior:
      if (scores[0] - scores[1]) >= tau → H = H_lo (usually 0)
      else → H = H_hi (e.g., 8)
    """
    def __init__(self, q: int = 12, K: int = 5, H_hi: int = 8, H_lo: int = 0, score_margin_thresh: float = 0.06):
        super().__init__(q=q, K=K, H=H_hi)
        self.H_hi = int(H_hi)
        self.H_lo = int(H_lo)
        self.tau = float(score_margin_thresh)

    def decide_H(self, scores: np.ndarray) -> int:
        if scores is None or len(scores) < 2:
            return self.H_lo  # confident by default if only one cand
        m = float(scores[0] - scores[1])
        return self.H_lo if m >= self.tau else self.H_hi

class BudgetPolicy(FixedMarginPolicy):
    """
    Budget-aware heuristic policy.

    Config keys under policy:
      algorithm: "budget"
      target_bpp: float           # desired overall bpp (guidance)
      q_bounds: [min_q, max_q]    # quant step bounds
      K_bounds: [min_K, max_K]    # candidate count bounds
      h_low: int                  # H when confident
      h_high: int                 # H when not confident
      score_margin_thresh: float  # margin threshold for H
    """
    def __init__(
        self,
        target_bpp: float = 0.7,
        q_bounds: tuple[int, int] = (10, 20),
        K_bounds: tuple[int, int] = (3, 6),
        h_low: int = 0,
        h_high: int = 8,
        score_margin_thresh: float = 0.06,
        default_q: int = 12,
        default_K: int = 5,
    ):
        super().__init__(q=default_q, K=default_K, H_hi=h_high, H_lo=h_low, score_margin_thresh=score_margin_thresh)
        self.tgt = float(target_bpp)
        self.q_lo, self.q_hi = int(q_bounds[0]), int(q_bounds[1])
        self.K_lo, self.K_hi = int(K_bounds[0]), int(K_bounds[1])

    @staticmethod
    def _safe_complexity(feats: np.ndarray) -> float:
        if feats is None or feats.size == 0:
            return 0.5
        f = np.asarray(feats, dtype=np.float32).flatten()
        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        a = np.abs(f)
        p50 = float(np.percentile(a, 50))
        p90 = float(np.percentile(a, 90))
        comp = 0.7 * p90 + 0.3 * p50
        comp = 1.0 - np.exp(-3.0 * comp)
        return float(np.clip(comp, 0.0, 1.0))

    def act(self, features: np.ndarray) -> Dict[str, int]:
        comp = self._safe_complexity(features)

        # Map complexity → q (higher comp ⇒ lower q)
        q_span = max(1, self.q_hi - self.q_lo)
        q = int(round(self.q_hi - comp * q_span))
        q = int(np.clip(q, self.q_lo, self.q_hi))

        # Map complexity → K
        K = int(round(self.K_lo + comp * (self.K_hi - self.K_lo)))
        K = int(np.clip(K, self.K_lo, self.K_hi))

        # H will be decided later via decide_H(scores) if available
        return {"q": q, "K": K, "H": self.H}

# -----------------------------
# RL Policy (TorchScript)
# -----------------------------

class RLPolicyTorch(Policy):
    """
    TorchScript policy that maps feature vector -> logits over bins for (q, K, H).
    Expects the same bins as rl.py:
      Q_BINS = 8..32 step 2
      K_BINS = [1,2,3,4,5,6]
      H_BINS = [0,8]
    """
    def __init__(self, script_path: str, fallback_q: int = 12, fallback_K: int = 5, fallback_H: int = 8, device: str = "cpu"):
        super().__init__(q=fallback_q, K=fallback_K, H=fallback_H)
        try:
            import torch  # defer import so non-RL users don't need torch
        except Exception as e:
            raise RuntimeError("RLPolicyTorch requires PyTorch. Install torch or switch policy.algorithm.") from e
        self.torch = torch
        self.device = device
        self.model = torch.jit.load(script_path, map_location=device).eval()
        self.Q = np.array(list(range(8, 33, 2)), dtype=np.int32)
        self.Ks = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
        self.Hs = np.array([0, 8], dtype=np.int32)

    def act(self, features: np.ndarray) -> Dict[str, int]:
        x = self.torch.as_tensor(features[None, ...], dtype=self.torch.float32, device=self.device)
        with self.torch.no_grad():
            lq, lk, lh, _ = self.model(x)
        qi = int(self.torch.argmax(lq, dim=-1).item())
        ki = int(self.torch.argmax(lk, dim=-1).item())
        hi = int(self.torch.argmax(lh, dim=-1).item())
        q = int(self.Q[np.clip(qi, 0, len(self.Q) - 1)])
        K = int(self.Ks[np.clip(ki, 0, len(self.Ks) - 1)])
        H = int(self.Hs[np.clip(hi, 0, len(self.Hs) - 1)])
        return {"q": q, "K": K, "H": H}

# -----------------------------
# Loaders
# -----------------------------

def load_policy(cfg: Dict[str, Any] | None):
    ec = cfg.get("encoder", {}) if cfg else {}
    pc = cfg.get("policy", {}) if cfg else {}

    algo = str(pc.get("algorithm", "fixed_margin")).lower()
    q = int(ec.get("default_q", 12))
    K = int(ec.get("default_K", 5))
    H = int(ec.get("default_H", 8))

    if algo == "fixed_margin":
        return FixedMarginPolicy(
            q=q,
            K=K,
            H_hi=H,
            H_lo=int(pc.get("h_low", 0)),  # allow override if present
            score_margin_thresh=float(pc.get("score_margin_thresh", 0.06)),
        )

    if algo == "budget":
        return BudgetPolicy(
            target_bpp=float(pc.get("target_bpp", 0.7)),
            q_bounds=tuple(pc.get("q_bounds", [10, 20])),
            K_bounds=tuple(pc.get("K_bounds", [3, 6])),
            h_low=int(pc.get("h_low", 0)),
            h_high=int(pc.get("h_high", H)),
            score_margin_thresh=float(pc.get("score_margin_thresh", 0.06)),
            default_q=q,
            default_K=K,
        )

    if algo == "rl":
        script = pc.get("script", None)
        if not script:
            raise ValueError("policy.algorithm=rl requires policy.script=<TorchScript path>")
        device = pc.get("device", "cpu")
        return RLPolicyTorch(script_path=str(script), fallback_q=q, fallback_K=K, fallback_H=H, device=device)

    # Fallback: fixed params
    return Policy(q=q, K=K, H=H)