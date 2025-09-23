from __future__ import annotations
import math, gc, os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
from ambi.io import load_image_rgb, rgb_to_ycbcr, BitWriter, fixed_quadtree_leaves
from ambi.core import quantize, dequantize, hash_prefix
from ambi.features import block_features
from ambi.models import DeterministicPrior
from ambi.partition import dynamic_quadtree_leaves
from ambi.compress import choose_compressor, COMP_NONE, COMP_ZLIB
from ambi.distortion import ssim, ms_ssim

if os.environ.get("AMBI_NOPROGRESS", "") == "1":
    class _NoTqdm:
        def __init__(self, *a, **k): pass
        def __enter__(self, *a, **k): return self
        def __exit__(self, exc_type, exc, tb): return False
        def update(self, *a, **k): pass
        def set_postfix_str(self, *a, **k): pass
    def tqdm(*a, **k):
        return _NoTqdm()
else:
    from tqdm.auto import tqdm

Q_BINS = list(range(8, 33, 2))
K_BINS = [1, 2, 3, 4, 5, 6]
H_BINS = [0, 8]

def _sobel_grad_mag(x: np.ndarray) -> np.ndarray:
    H, W = x.shape
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
    ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
    xp = np.pad(x, ((1,1),(1,1)), mode="edge").astype(np.float32)
    gx = (kx[0,0]*xp[0:H,0:W] + kx[0,1]*xp[0:H,1:W+1] + kx[0,2]*xp[0:H,2:W+2] +
          kx[1,0]*xp[1:H+1,0:W] + kx[1,1]*xp[1:H+1,1:W+1] + kx[1,2]*xp[1:H+1,2:W+2] +
          kx[2,0]*xp[2:H+2,0:W] + kx[2,1]*xp[2:H+2,1:W+1] + kx[2,2]*xp[2:H+2,2:W+2])
    gy = (ky[0,0]*xp[0:H,0:W] + ky[0,1]*xp[0:H,1:W+1] + ky[0,2]*xp[0:H,2:W+2] +
          ky[1,0]*xp[1:H+1,0:W] + ky[1,1]*xp[1:H+1,1:W+1] + ky[1,2]*xp[1:H+1,2:W+2] +
          ky[2,0]*xp[2:H+2,0:W] + ky[2,1]*xp[2:H+2,1:W+1] + ky[2,2]*xp[2:H+2,2:W+2])
    return np.sqrt(gx * gx + gy * gy, dtype=np.float32)

def _to_eval_space(img: np.ndarray, on: str = "y") -> np.ndarray:
    on = (on or "y").lower()
    if on.startswith("y"):
        r, g, b = img[...,0], img[...,1], img[...,2]
        y = (0.299*r + 0.587*g + 0.114*b).astype(np.float32)
        return y
    return img.astype(np.float32)

def gmsd(x: np.ndarray, y: np.ndarray, on: str = "y", c: float = 170.0) -> float:
    assert x.shape[:2] == y.shape[:2]
    H, W = x.shape[:2]
    if H == 0 or W == 0:
        return 0.0
    X = np.clip(x, 0.0, 1.0).astype(np.float32)
    Y = np.clip(y, 0.0, 1.0).astype(np.float32)
    if (on or "y").lower().startswith("y"):
        Xy = _to_eval_space(X, "y")
        Yy = _to_eval_space(Y, "y")
    else:
        Xy = np.sqrt((X[...,0]**2 + X[...,1]**2 + X[...,2]**2)/3.0, dtype=np.float32)
        Yy = np.sqrt((Y[...,0]**2 + Y[...,1]**2 + Y[...,2]**2)/3.0, dtype=np.float32)
    gm_x = _sobel_grad_mag(Xy)
    gm_y = _sobel_grad_mag(Yy)
    c = float(c)
    gms = (2.0 * gm_x * gm_y + c) / (gm_x * gm_x + gm_y * gm_y + c)
    gms = np.clip(gms, 0.0, 1.0)
    return float(gms.std())

def aug_hflip(img: np.ndarray) -> np.ndarray:
    return img[:, ::-1, :].copy()

def aug_vflip(img: np.ndarray) -> np.ndarray:
    return img[::-1, :, :].copy()

def aug_rot90(img: np.ndarray, k: int = 1) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(img, k, axes=(0, 1)))

def make_augs(img: np.ndarray, kinds: List[str], per_image: int) -> List[np.ndarray]:
    pool: List[np.ndarray] = []
    if "hflip" in kinds:
        pool.append(aug_hflip(img))
    if "vflip" in kinds:
        pool.append(aug_vflip(img))
    if "rot90" in kinds:
        pool.append(aug_rot90(img, 1))
        pool.append(aug_rot90(img, 2))
        pool.append(aug_rot90(img, 3))
    if per_image <= 0 or len(pool) == 0:
        return []
    return pool[:per_image]

@dataclass
class EnvCfg:
    min_block: int = 32
    max_block: int = 32
    use_quadtree: bool = False
    split_var_thresh: Optional[float] = None
    split_grad_thresh: Optional[float] = None
    target_bpp: float = 0.6
    version: int = 3
    compression: Dict[str, Any] = None
    lamb_r: float = 1.0
    lamb_d: float = 200.0
    lamb_ssim: float = 50.0
    lamb_msssim: float = 50.0
    lamb_gmsd: float = 0.0
    lamb_t: float = 0.0
    control_split: bool = False
    split_bit_cost: float = 1.0
    msssim_on: str = "y"
    ds_factor: float = 1.0
    msssim_pow: float = 1.0
    obs_dim: Optional[int] = None

class AMBIEnv:
    def __init__(self, img: np.ndarray, cfg: EnvCfg):
        self.cfg = cfg
        self.ycc = rgb_to_ycbcr(img)
        h, w, _ = self.ycc.shape
        self.compressor = choose_compressor({"encoder": {"compression": (cfg.compression or {"type": "zlib", "level": 6})}})
        self.have_comp = (cfg.version >= 3)
        self.prior = DeterministicPrior(K=max(K_BINS))
        if cfg.control_split:
            mb = max(1, int(cfg.max_block))
            roots: List[Tuple[int,int,int,int]] = []
            for y in range(0, h, mb):
                for x in range(0, w, mb):
                    bw = min(mb, w - x)
                    bh = min(mb, h - y)
                    roots.append((x, y, bw, bh))
            self.stack = deque(roots[::-1])
            self._fixed_leaves: List[Tuple[int,int,int,int]] = []
        else:
            if cfg.use_quadtree and cfg.max_block > cfg.min_block:
                leaves = dynamic_quadtree_leaves(
                    None if self.ycc is None else self.ycc,
                    min_block=cfg.min_block,
                    max_block=cfg.max_block,
                    var_thresh=cfg.split_var_thresh,
                    grad_thresh=cfg.split_grad_thresh,
                )
            else:
                leaves = fixed_quadtree_leaves(w, h, cfg.min_block)
            self._fixed_leaves = leaves
            self.stack = deque(leaves[::-1])
        self.n_total = len(self.stack)
        self.done_flag = (self.n_total == 0)
        if cfg.obs_dim is None and not self.done_flag:
            x0, y0, bw0, bh0 = self.stack[-1]
            f0 = block_features(self.ycc[y0:y0+bh0, x0:x0+bw0, :]).astype(np.float32).ravel()
            self.obs_dim = int(f0.size)
        else:
            self.obs_dim = int(cfg.obs_dim) if cfg.obs_dim is not None else 64

    def _pad_or_trunc(self, feat: np.ndarray) -> np.ndarray:
        feat = np.asarray(feat, dtype=np.float32).ravel()
        d = self.obs_dim
        if feat.size == d:
            return feat
        if feat.size > d:
            return feat[:d]
        out = np.zeros((d,), dtype=np.float32)
        out[:feat.size] = feat
        return out

    def _current_block(self) -> Optional[Tuple[int,int,int,int]]:
        if not self.stack:
            return None
        return self.stack[-1]

    def reset(self) -> np.ndarray:
        h, w, _ = self.ycc.shape
        if self.cfg.control_split:
            mb = max(1, int(self.cfg.max_block))
            roots = []
            for y in range(0, h, mb):
                for x in range(0, w, mb):
                    bw = min(mb, w - x)
                    bh = min(mb, h - y)
                    roots.append((x, y, bw, bh))
            self.stack = deque(roots[::-1])
        else:
            self.stack = deque(self._fixed_leaves[::-1])
        self.done_flag = (len(self.stack) == 0)
        return self._obs()

    def _obs(self) -> np.ndarray:
        if self.done_flag or not self.stack:
            return np.zeros((self.obs_dim,), np.float32)
        x, y, bw, bh = self.stack[-1]
        b = self.ycc[y:y+bh, x:x+bw, :]
        return self._pad_or_trunc(block_features(b))

    @staticmethod
    def _soft_qweight(x: float,
                      q1: float, med: float, q3: float,
                      w_q1: float, w_med: float, w_q3: float,
                      invert: bool = False) -> float:
        eps = 1e-6
        l1 = max(med - q1, eps)
        l2 = max(q3 - med, eps)
        if x <= med:
            s_q1  = max(0.0, (med - x) / l1)
            s_med = max(0.0, 1.0 - (med - x) / l1)
            s_q3  = 0.0
        else:
            s_q1  = 0.0
            s_med = max(0.0, 1.0 - (x - med) / l2)
            s_q3  = max(0.0, (x - med) / l2)
        if invert:
            s_q1, s_q3 = s_q3, s_q1
        S = s_q1 + s_med + s_q3
        if S <= eps:
            return float(w_med)
        s_q1 /= S; s_med /= S; s_q3 /= S
        return float(w_q1 * s_q1 + w_med * s_med + w_q3 * s_q3)
    
    @staticmethod
    def _tri_memberships(x: float, q1: float, med: float, q3: float) -> Tuple[float, float, float]:
        eps = 1e-6
        l1 = max(med - q1, eps)
        l2 = max(q3 - med, eps)
        if x <= med:
            s_low  = max(0.0, (med - x) / l1)
            s_mid  = max(0.0, 1.0 - (med - x) / l1)
            s_high = 0.0
        else:
            s_low  = 0.0
            s_mid  = max(0.0, 1.0 - (x - med) / l2)
            s_high = max(0.0, (x - med) / l2)
        S = s_low + s_mid + s_high
        if S <= eps:
            return 0.0, 1.0, 0.0
        return s_low / S, s_mid / S, s_high / S

    def _encode_reward(self, x, y, bw, bh, q, K, H) -> Tuple[float, Dict[str, float]]:
        b = self.ycc[y:y+bh, x:x+bw, :]
        qblk = quantize(b, q)
        coarse = dequantize(qblk, q)
        cands, _scores = self.prior.propose(coarse, K=K)
        errs = [float(np.mean((np.clip(c, 0, 1) - np.clip(b, 0, 1)) ** 2)) for c in cands]
        best = int(np.argmin(errs))
        if H == 0:
            hp = 0; refine_flag = 0; refine_idx = 0
        else:
            hp = int(hash_prefix(cands[best], H))
            survivors = [j for j, c in enumerate(cands) if hash_prefix(c, H) == hp]
            if len(survivors) <= 1:
                refine_flag = 0; refine_idx = 0
            else:
                refine_flag = 1
                refine_idx = int(survivors.index(best))
        bwb = BitWriter()
        bwb.write_varint(x); bwb.write_varint(y)
        bwb.write_varint(bw); bwb.write_varint(bh)
        bwb.write_varint(q); bwb.write_varint(K); bwb.write_varint(H)
        bwb.write_varint(hp)
        bwb.write_varint(refine_flag)
        if refine_flag:
            bwb.write_varint(refine_idx)
        raw = qblk.astype(np.int16).tobytes(order="C")
        if self.have_comp:
            comp_id, payload, raw_len = self.compressor.compress(raw)
            bwb.write_varint(int(comp_id))
            if comp_id == COMP_NONE:
                bwb.write_varint(raw_len); bwb.write_bytes(payload)
            elif comp_id == COMP_ZLIB:
                bwb.write_varint(len(payload)); bwb.write_varint(raw_len); bwb.write_bytes(payload)
            else:
                bwb.write_varint(raw_len); bwb.write_bytes(raw)
        else:
            bwb.write_varint(len(raw)); bwb.write_bytes(raw)
        rec_len_bytes = len(bwb.getvalue())
        bpp = (rec_len_bytes * 8.0) / float(bw * bh)
        mse = float(np.mean((np.clip(cands[best], 0, 1) - np.clip(b, 0, 1)) ** 2))
        psnr_val = 10.0 * math.log10(1.0 / max(mse, 1e-12))
        on = (self.cfg.msssim_on or "y").lower()
        if self.cfg.ds_factor and self.cfg.ds_factor != 1.0:
            sf = float(self.cfg.ds_factor)
            Hs = max(1, int(round(bh * sf)))
            Ws = max(1, int(round(bw * sf)))
            ref_eval = np.clip(b, 0, 1)
            rec_eval = np.clip(cands[best], 0, 1)
            ref_eval = np.clip(torch.nn.functional.interpolate(
                torch.from_numpy(ref_eval).permute(2,0,1).unsqueeze(0),
                size=(Hs, Ws), mode="area").squeeze(0).permute(1,2,0).numpy(), 0, 1)
            rec_eval = np.clip(torch.nn.functional.interpolate(
                torch.from_numpy(rec_eval).permute(2,0,1).unsqueeze(0),
                size=(Hs, Ws), mode="area").squeeze(0).permute(1,2,0).numpy(), 0, 1)
        else:
            ref_eval = np.clip(b, 0, 1)
            rec_eval = np.clip(cands[best], 0, 1)
        ssim_val = float(ssim(ref_eval, rec_eval, on=on))
        msssim_val = float(ms_ssim(ref_eval, rec_eval, on=on))
        gmsd_val = float(gmsd(ref_eval, rec_eval, on=on))
        a_bpp = float(getattr(self, "_alpha_bpp", 1.0))
        a_psnr = float(getattr(self, "_alpha_psnr", 1.0))
        a_mss = float(getattr(self, "_alpha_msssim", 1.0))
        
        bq1, bmed, bq3 = getattr(self.cfg, "_bpp_cuts", (bpp, bpp, bpp))
        pq1, pmed, pq3 = getattr(self.cfg, "_psnr_cuts", (psnr_val, psnr_val, psnr_val))
        wb_q1, wb_med, wb_q3 = getattr(self.cfg, "_bpp_weights", (1.0, 1.0, 1.0))
        wp_q1, wp_med, wp_q3 = getattr(self.cfg, "_psnr_weights", (1.0, 1.0, 1.0))
        w_bpp  = self._soft_qweight(bpp,      bq1, bmed, bq3, wb_q1, wb_med, wb_q3, invert=False)
        w_psnr = self._soft_qweight(psnr_val, pq1, pmed, pq3, wp_q1, wp_med, wp_q3, invert=True)
        lb, mb, hb = self._tri_memberships(bpp, bq1, bmed, bq3)
        lp, mp, hp = self._tri_memberships(psnr_val, pq1, pmed, pq3)
        a_mismatch = float(getattr(self, "_alpha_mismatch", 0.0))
        mismatch_pen = a_mismatch * hb * lp
        R = (a_bpp * w_bpp * bpp) - (a_psnr * w_psnr * psnr_val) - (a_mss * msssim_val) + mismatch_pen
        reward = -float(R)
        info = {
            "bpp": bpp, "mse": mse, "ssim": ssim_val, "msssim": msssim_val, "gmsd": gmsd_val,
            "q": Q_BINS.index(q) if q in Q_BINS else q,
            "K": K, "H": H, "split": 0,
            "reward": reward, "score": -reward,
            "psnr": psnr_val,
        }
        return reward, info

    def _split_reward(self, x, y, bw, bh) -> Tuple[float, Dict[str, float]]:
        bpp = float(self.cfg.split_bit_cost) / float(max(1, bw * bh))
        WBPP = float(getattr(self, "_WBPP", 1.0))
        a_bpp = float(getattr(self, "_alpha_bpp", 1.0))
        R = a_bpp * WBPP * bpp
        reward = -float(R)
        info = {
            "bpp": bpp, "mse": 0.0, "ssim": 1.0, "msssim": 1.0, "gmsd": 0.0,
            "q": 0, "K": 0, "H": 0, "split": 1,
            "reward": reward, "score": -reward,
            "psnr": 120.0,
        }
        return reward, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done_flag or not self.stack:
            return self._obs(), 0.0, True, {}
        x, y, bw, bh = self.stack.pop()
        if self.cfg.control_split:
            split, q_idx, K_idx, H_idx = action
            split = int(split)
            q = Q_BINS[int(q_idx)]
            K = K_BINS[int(K_idx)]
            H = H_BINS[int(H_idx)]
            if split == 1 and (bw > self.cfg.min_block or bh > self.cfg.min_block):
                cw = max(self.cfg.min_block, bw // 2)
                ch = max(self.cfg.min_block, bh // 2)
                x1, y1 = x, y
                x2, y2 = x + cw, y + ch
                blocks: List[Tuple[int,int,int,int]] = []
                blocks.append((x1, y1, min(cw, bw), min(ch, bh)))
                if x2 < x + bw:  blocks.append((x2, y1, bw - (x2 - x), min(ch, bh)))
                if y2 < y + bh:  blocks.append((x1, y2, min(cw, bw), bh - (y2 - y)))
                if (x2 < x + bw) and (y2 < y + bh):
                    blocks.append((x2, y2, bw - (x2 - x), bh - (y2 - y)))
                for blk in reversed(blocks):
                    self.stack.append(blk)
                reward, info = self._split_reward(x, y, bw, bh)
                done = (len(self.stack) == 0)
                self.done_flag = done
                return self._obs(), reward, done, info
            else:
                reward, info = self._encode_reward(x, y, bw, bh, q, K, H)
                done = (len(self.stack) == 0)
                self.done_flag = done
                return self._obs(), reward, done, info
        else:
            q_idx, K_idx, H_idx = action
            q = Q_BINS[int(q_idx)]
            K = K_BINS[int(K_idx)]
            H = H_BINS[int(H_idx)]
            reward, info = self._encode_reward(x, y, bw, bh, q, K, H)
            done = (len(self.stack) == 0)
            self.done_flag = done
            return self._obs(), reward, done, info

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, control_split: bool, hid: int = 128):
        super().__init__()
        self.control_split = bool(control_split)
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
        )
        if self.control_split:
            self.pi_split = nn.Linear(hid, 1)
        self.pi_q = nn.Linear(hid, len(Q_BINS))
        self.pi_k = nn.Linear(hid, len(K_BINS))
        self.pi_h = nn.Linear(hid, len(H_BINS))
        self.v = nn.Linear(hid, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        if self.control_split:
            logit_split = self.pi_split(h).squeeze(-1)
            return logit_split, self.pi_q(h), self.pi_k(h), self.pi_h(h), self.v(h).squeeze(-1)
        else:
            return None, self.pi_q(h), self.pi_k(h), self.pi_h(h), self.v(h).squeeze(-1)

@dataclass
class PPOCfg:
    lr: float = 3e-4
    epochs: int = 3
    steps_per_iter: int = 4096
    minibatch: int = 256
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    max_iters: int = 50
    device: str = "cpu"
    env_workers: Optional[int] = None

class PPOTrainer:
    def __init__(self, obs_dim: int, cfg: PPOCfg, control_split: bool):
        self.cfg = cfg
        self.obs_dim = int(obs_dim)
        self.control_split = bool(control_split)
        self.net = PolicyNet(self.obs_dim, self.control_split).to(cfg.device)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.lr)

    def _gather(self, envs: List[AMBIEnv], steps: int):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        info_buf: List[Dict[str, Any]] = []
        n_envs = len(envs)
        n_workers = int(self.cfg.env_workers or n_envs)
        states = [env.reset() for env in envs]
        states_t = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.cfg.device)
        with ThreadPoolExecutor(max_workers=n_workers) as pool, \
             tqdm(total=steps, desc="RL rollout", unit="step", leave=False) as pbar:
            def _step_one(i_env: int, action_tuple):
                ns, r, d, info = envs[i_env].step(action_tuple)
                if d:
                    ns = envs[i_env].reset()
                return i_env, ns, r, d, info
            for _ in range(steps):
                if self.control_split:
                    logit_split, lq, lk, lh, v = self.net.forward(states_t)
                    pi_split = torch.distributions.Bernoulli(logits=logit_split)
                    piq = torch.distributions.Categorical(logits=lq)
                    pik = torch.distributions.Categorical(logits=lk)
                    pih = torch.distributions.Categorical(logits=lh)
                    a_split = pi_split.sample()
                    aq = piq.sample()
                    ak = pik.sample()
                    ah = pih.sample()
                    keep_mask = (1.0 - a_split).detach()
                    logp = (pi_split.log_prob(a_split)
                            + (piq.log_prob(aq) + pik.log_prob(ak) + pih.log_prob(ah)) * keep_mask)
                    split_np = a_split.detach().cpu().numpy().astype(np.int64)
                    q_idx = aq.detach().cpu().numpy().astype(np.int64)
                    k_idx = ak.detach().cpu().numpy().astype(np.int64)
                    h_idx = ah.detach().cpu().numpy().astype(np.int64)
                    logp_np = logp.detach().cpu().numpy()
                    vals = v.detach().cpu().numpy()
                    next_states = [None] * n_envs
                    rewards = np.zeros((n_envs,), dtype=np.float32)
                    dones = np.zeros((n_envs,), dtype=bool)
                    futures = [
                        pool.submit(_step_one, i, (int(split_np[i]), int(q_idx[i]), int(k_idx[i]), int(h_idx[i])))
                        for i in range(n_envs)
                    ]
                    for f in as_completed(futures):
                        i, ns, r, d, info = f.result()
                        next_states[i] = ns; rewards[i] = r; dones[i] = d; info_buf.append(info)
                    obs_buf.append(np.stack(states))
                    act_buf.append(np.stack([split_np, q_idx, k_idx, h_idx], axis=1))
                    logp_buf.append(logp_np)
                    val_buf.append(vals)
                    rew_buf.append(rewards)
                    done_buf.append(dones)
                else:
                    _none, lq, lk, lh, v = self.net.forward(states_t)
                    piq = torch.distributions.Categorical(logits=lq)
                    pik = torch.distributions.Categorical(logits=lk)
                    pih = torch.distributions.Categorical(logits=lh)
                    aq = piq.sample(); ak = pik.sample(); ah = pih.sample()
                    logp = piq.log_prob(aq) + pik.log_prob(ak) + pih.log_prob(ah)
                    q_idx = aq.detach().cpu().numpy().astype(np.int64)
                    k_idx = ak.detach().cpu().numpy().astype(np.int64)
                    h_idx = ah.detach().cpu().numpy().astype(np.int64)
                    logp_np = logp.detach().cpu().numpy()
                    vals = v.detach().cpu().numpy()
                    next_states = [None] * n_envs
                    rewards = np.zeros((n_envs,), dtype=np.float32)
                    dones = np.zeros((n_envs,), dtype=bool)
                    futures = [pool.submit(_step_one, i, (int(q_idx[i]), int(k_idx[i]), int(h_idx[i])))
                               for i in range(n_envs)]
                    for f in as_completed(futures):
                        i, ns, r, d, info = f.result()
                        next_states[i] = ns; rewards[i] = r; dones[i] = d; info_buf.append(info)
                    obs_buf.append(np.stack(states))
                    act_buf.append(np.stack([q_idx, k_idx, h_idx], axis=1))
                    logp_buf.append(logp_np)
                    val_buf.append(vals)
                    rew_buf.append(rewards)
                    done_buf.append(dones)
                states = next_states
                states_t = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.cfg.device)
                if info_buf and ("bpp" in info_buf[-1] or "mse" in info_buf[-1]):
                    last = info_buf[-1]
                    try:
                        psnr = last.get("psnr", 10.0 * math.log10(1.0 / max(last.get("mse", 1e-12), 1e-12)))
                        pbar.set_postfix_str(
                            f"split={last.get('split',0)}, bpp={last.get('bpp',0):.3f}, "
                            f"PSNR={psnr:.2f}, SSIM={last.get('ssim',0):.3f}, "
                            f"MS={last.get('msssim',0):.3f}, GMSD={last.get('gmsd',0):.4f}"
                        )
                    except Exception:
                        pass
                pbar.update(1)
        obs = torch.as_tensor(np.concatenate(obs_buf, axis=0), dtype=torch.float32, device=self.cfg.device)
        act = torch.as_tensor(np.concatenate(act_buf, axis=0), dtype=torch.long, device=self.cfg.device)
        logp_old = torch.as_tensor(np.concatenate(logp_buf, axis=0), dtype=torch.float32, device=self.cfg.device)
        val = torch.as_tensor(np.concatenate(val_buf, axis=0), dtype=torch.float32, device=self.cfg.device)
        rew = np.concatenate(rew_buf, axis=0)
        done = np.concatenate(done_buf, axis=0)
        adv = np.zeros_like(rew)
        lastgaelam = 0.0
        vals_np = val.detach().cpu().numpy()
        for t in reversed(range(len(rew))):
            nonterminal = 1.0 - done[t]
            nextv = vals_np[t+1] if t+1 < len(vals_np) else 0.0
            delta = rew[t] + self.cfg.gamma * nextv * nonterminal - vals_np[t]
            lastgaelam = delta + self.cfg.gamma * self.cfg.lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + vals_np
        adv = torch.as_tensor(adv, dtype=torch.float32, device=self.cfg.device)
        ret = torch.as_tensor(ret, dtype=torch.float32, device=self.cfg.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return obs, act, logp_old, adv, ret, info_buf

    def update(self, obs, act, logp_old, adv, ret):
        cfg = self.cfg
        n = obs.shape[0]
        idx = np.arange(n)
        with tqdm(total=cfg.epochs, desc="PPO epochs", unit="epoch", leave=False) as epbar:
            for e in range(cfg.epochs):
                np.random.shuffle(idx)
                num_batches = int(math.ceil(n / cfg.minibatch))
                with tqdm(total=num_batches, desc=f"epoch {e+1}/{cfg.epochs} (minibatches)", unit="mb", leave=False) as mbbar:
                    s = 0
                    while s < n:
                        mb = idx[s:s+cfg.minibatch]
                        mb_obs = obs[mb]
                        mb_logp_old = logp_old[mb]
                        mb_adv = adv[mb]
                        mb_ret = ret[mb]
                        if self.control_split:
                            mb_act = act[mb]
                            logit_split, lq, lk, lh, v = self.net.forward(mb_obs)
                            pi_split = torch.distributions.Bernoulli(logits=logit_split)
                            piq = torch.distributions.Categorical(logits=lq)
                            pik = torch.distributions.Categorical(logits=lk)
                            pih = torch.distributions.Categorical(logits=lh)
                            a_split = mb_act[:, 0].float()
                            a_q = mb_act[:, 1]
                            a_k = mb_act[:, 2]
                            a_h = mb_act[:, 3]
                            keep_mask = (1.0 - a_split)
                            logp = (pi_split.log_prob(a_split)
                                    + (piq.log_prob(a_q) + pik.log_prob(a_k) + pih.log_prob(a_h)) * keep_mask)
                            ratio = torch.exp(logp - mb_logp_old)
                            surr1 = ratio * mb_adv
                            surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * mb_adv
                            policy_loss = -torch.min(surr1, surr2).mean()
                            value_loss = ((v - mb_ret) ** 2).mean()
                            entropy = (pi_split.entropy() + (piq.entropy() + pik.entropy() + pih.entropy()) * keep_mask).mean()
                            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                        else:
                            mb_act = act[mb]
                            _none, lq, lk, lh, v = self.net.forward(mb_obs)
                            piq = torch.distributions.Categorical(logits=lq)
                            pik = torch.distributions.Categorical(logits=lk)
                            pih = torch.distributions.Categorical(logits=lh)
                            logp = piq.log_prob(mb_act[:, 0]) + pik.log_prob(mb_act[:, 1]) + pih.log_prob(mb_act[:, 2])
                            ratio = torch.exp(logp - mb_logp_old)
                            surr1 = ratio * mb_adv
                            surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * mb_adv
                            policy_loss = -torch.min(surr1, surr2).mean()
                            value_loss = ((v - mb_ret) ** 2).mean()
                            entropy = (piq.entropy() + pik.entropy() + pih.entropy()).mean()
                            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                        self.opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                        self.opt.step()
                        mbbar.set_postfix_str(f"loss={loss.item():.4f}, V={v.mean().item():.3f}")
                        mbbar.update(1)
                        s += cfg.minibatch
                epbar.update(1)

    def save_torchscript(self, path: Path):
        self.net.eval()
        example = torch.randn(1, self.obs_dim, device=self.cfg.device, dtype=torch.float32)
        scripted = torch.jit.trace(self.net, example)
        path.parent.mkdir(parents=True, exist_ok=True)
        scripted.save(str(path))

def list_image_paths(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [p for p in sorted(root.rglob("*")) if p.suffix.lower() in exts]

class ImageBatcher:
    def __init__(
        self,
        paths: List[Path],
        mode: str = "count",
        batch_size: int = 32,
        mem_fraction: float = 0.75,
        safety_fraction: float = 0.05,
        aug_per_image: int = 3,
        aug_kinds: Optional[List[str]] = None,
        shuffle_seed: int = 0,
        cached_imgs: Optional[List[np.ndarray]] = None
    ):
        self.paths = list(paths)
        self.mode = mode
        self.batch_size = int(max(1, batch_size))
        self.mem_fraction = float(max(0.05, min(0.95, mem_fraction)))
        self.safety_fraction = float(max(0.0, min(0.25, safety_fraction)))
        self.aug_per_image = int(max(0, aug_per_image))
        self.aug_kinds = list(aug_kinds) if aug_kinds is not None else ["hflip", "vflip", "rot90"]
        self.shuffle_seed = int(shuffle_seed)
        self._i = 0
        self._cache: Optional[List[np.ndarray]] = None
        if cached_imgs is not None:
            self._cache = list(cached_imgs)
        elif self.mode.lower() == "eager":
            rng = np.random.default_rng(self.shuffle_seed)
            imgs: List[np.ndarray] = []
            for p in self.paths:
                img = load_image_rgb(p)
                imgs.append(img)
                augs = make_augs(img, self.aug_kinds, self.aug_per_image)
                imgs.extend(augs)
            if len(imgs) > 1:
                rng.shuffle(imgs)
            self._cache = imgs

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self) -> List[np.ndarray]:
        if self._cache is not None:
            if self._i >= len(self._cache):
                raise StopIteration
            j = min(self._i + self.batch_size, len(self._cache))
            batch = self._cache[self._i:j]
            self._i = j
            return batch
        if self._i >= len(self.paths):
            raise StopIteration
        if self.mode == "count":
            j = min(self._i + self.batch_size, len(self.paths))
            batch_paths = self.paths[self._i:j]
            imgs: List[np.ndarray] = []
            for p in batch_paths:
                img = load_image_rgb(p)
                imgs.append(img)
                augs = make_augs(img, self.aug_kinds, self.aug_per_image)
                imgs.extend(augs)
            self._i = j
            if not imgs:
                raise StopIteration
            if len(imgs) > 1:
                rng = np.random.default_rng(self.shuffle_seed)
                rng.shuffle(imgs)
            return imgs
        avail0 = psutil.virtual_memory().available
        budget = max(128*1024*1024, int(avail0 * self.mem_fraction))
        imgs: List[np.ndarray] = []
        used = 0
        while self._i < len(self.paths):
            p = self.paths[self._i]
            img = load_image_rgb(p)
            est = int(img.nbytes)
            vm = psutil.virtual_memory()
            soft_guard = int(vm.available * (1.0 - self.safety_fraction))
            need = est
            augs = make_augs(img, self.aug_kinds, self.aug_per_image)
            for a in augs:
                need += int(a.nbytes)
            if used + need > budget or vm.available < soft_guard:
                if len(imgs) == 0:
                    imgs.append(img)
                    imgs.extend(augs[:max(0, (budget - est) // max(1, est))])
                    self._i += 1
                break
            imgs.append(img)
            imgs.extend(augs)
            used += need
            self._i += 1
        if not imgs:
            raise StopIteration
        if len(imgs) > 1:
            rng = np.random.default_rng(self.shuffle_seed)
            rng.shuffle(imgs)
        return imgs

def preload_all_images(paths: List[Path], aug_kinds: List[str], aug_per_image: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    imgs: List[np.ndarray] = []
    for p in paths:
        img = load_image_rgb(p)
        imgs.append(img)
        augs = make_augs(img, aug_kinds, aug_per_image)
        imgs.extend(augs)
    if len(imgs) > 1:
        rng.shuffle(imgs)
    return imgs

def build_envs(imgs: List[np.ndarray], cfg: EnvCfg, n_envs: int) -> List[AMBIEnv]:
    probe = AMBIEnv(imgs[0], cfg)
    shared_dim = probe.obs_dim
    cfg2 = replace(cfg, obs_dim=shared_dim)
    envs = [AMBIEnv(imgs[i % len(imgs)], cfg2) for i in range(n_envs)]
    for e in envs:
        for name, default in (
            ("_WBPP", 1.0),
            ("_WPSNR", 1.0),
            ("_alpha_bpp", 1.0),
            ("_alpha_psnr", 1.0),
            ("_alpha_msssim", 1.0),
            ("_bpp_cuts", None),
            ("_psnr_cuts", None),
            ("_bpp_weights", (1.0, 1.0, 1.0)),
            ("_psnr_weights", (1.0, 1.0, 1.0)),
            ("_alpha_mismatch", 0.0),
        ):
            setattr(e, name, getattr(cfg, name, default))
    return envs

@dataclass
class LoadingCfg:
    mode: str = "eager"
    lazy_by: str = "count"
    batch_size: int = 32
    mem_fraction: float = 0.75
    iters_per_batch: int = 1
    aug_per_image: int = 3
    aug_kinds: Optional[List[str]] = None
    shuffle_seed: int = 0

@dataclass
class EarlyStopCfg:
    enabled: bool = False
    patience: int = 10
    min_delta_bpp: float = 0.0
    min_delta_db: float = 0.0
    start_after: int = 0

class EarlyStopper:
    def __init__(self, cfg: EarlyStopCfg):
        self.cfg = cfg
        self.best_bpp = float("inf")
        self.best_db = float("-inf")
        self.wait = 0

    def update(self, epoch_idx: int, bpp: float, db: float):
        if not self.cfg.enabled or epoch_idx < self.cfg.start_after:
            if bpp < self.best_bpp: self.best_bpp = bpp
            if db  > self.best_db:  self.best_db  = db
            self.wait = 0
            return False, True, True
        bpp_imp = (self.best_bpp - bpp) >= self.cfg.min_delta_bpp
        db_imp  = (db - self.best_db)   >= self.cfg.min_delta_db
        if bpp_imp: self.best_bpp = bpp
        if db_imp:  self.best_db  = db
        if bpp_imp or db_imp:
            self.wait = 0
        else:
            self.wait += 1
        return (self.wait >= self.cfg.patience), bpp_imp, db_imp

def train_rl(
    data_root: Path,
    out_model: Path,
    yaml_cfg: Dict[str, Any],
    ppo_cfg: PPOCfg = PPOCfg(),
    n_envs: int = 4,
    iters: int = 10,
) -> None:
    enc = yaml_cfg.get("encoder", {}) if yaml_cfg else {}
    fmt = yaml_cfg.get("format", {}) if yaml_cfg else {}
    split = enc.get("split", {}) if isinstance(enc.get("split", {}), dict) else {}
    rl = yaml_cfg.get("rl", {}) if yaml_cfg else {}
    w = rl.get("weights", {}) if isinstance(rl.get("weights", {}), dict) else {}
    load = rl.get("loading", {}) if isinstance(rl.get("loading", {}), dict) else {}
    env_cfg = EnvCfg(
        min_block=int(enc.get("min_block", enc.get("block_size", 32))),
        max_block=int(enc.get("max_block", enc.get("block_size", 32))),
        use_quadtree=int(enc.get("max_block", 32)) > int(enc.get("min_block", 32)),
        split_var_thresh=float(split.get("var_thresh", 0.0025)) if split else None,
        split_grad_thresh=float(split.get("grad_thresh", 0.02)) if split else None,
        target_bpp=float(yaml_cfg.get("policy", {}).get("target_bpp", 0.6)),
        version=int(fmt.get("version", 3)),
        compression=enc.get("compression", {"type": "zlib", "level": 6}),
        lamb_r=float(w.get("rate", rl.get("lamb_r", 1.0))),
        lamb_d=float(w.get("mse", rl.get("lamb_d", 200.0))),
        lamb_ssim=float(w.get("ssim", rl.get("lamb_ssim", 50.0))),
        lamb_msssim=float(w.get("msssim", rl.get("lamb_msssim", 50.0))),
        lamb_gmsd=float(w.get("gmsd", rl.get("lamb_gmsd", 0.0))),
        lamb_t=float(w.get("rt", rl.get("lamb_t", 0.0))),
        control_split=bool(rl.get("control_split", False)),
        split_bit_cost=float(rl.get("split_bit_cost", 1.0)),
        msssim_on=str(rl.get("msssim_on", "y")),
        ds_factor=float(rl.get("ds_factor", 1.0)),
        msssim_pow=float(rl.get("msssim_pow", 1.0)),
        obs_dim=None,
    )
    bpp_q1  = float(w.get("bpp_q1", 1/3))
    bpp_med = float(w.get("bpp_med", 1/3))
    bpp_q3  = float(w.get("bpp_q3", 1/3))
    psnr_q1  = float(w.get("psnr_q1", 1/3))
    psnr_med = float(w.get("psnr_med", 1/3))
    psnr_q3  = float(w.get("psnr_q3", 1/3))
    alpha_bpp_base    = float(w.get("alpha_bpp", 1.0))
    alpha_psnr_base   = float(w.get("alpha_psnr", 1.0))
    alpha_msssim      = float(w.get("alpha_msssim", 1.0))
    alpha_mismatch = float(w.get("alpha_mismatch", 0.0))
    WBPP  = bpp_q1 + bpp_med + bpp_q3
    WPSNR = psnr_q1 + psnr_med + psnr_q3
    paths = list_image_paths(data_root)
    if not paths:
        raise ValueError("No images found for RL dataset.")
    probe_img = load_image_rgb(paths[0])
    env_probe = AMBIEnv(probe_img, env_cfg)
    setattr(env_cfg, "_WBPP", WBPP)
    setattr(env_cfg, "_WPSNR", WPSNR)
    setattr(env_cfg, "_alpha_bpp", alpha_bpp_base)
    setattr(env_cfg, "_alpha_psnr", alpha_psnr_base)
    setattr(env_cfg, "_alpha_msssim", alpha_msssim)
    setattr(env_cfg, "_alpha_mismatch", alpha_mismatch)
    setattr(env_cfg, "_bpp_weights", (bpp_q1, bpp_med, bpp_q3))
    setattr(env_cfg, "_psnr_weights", (psnr_q1, psnr_med, psnr_q3))
    obs_dim = int(env_probe.obs_dim)
    trainer = PPOTrainer(obs_dim, ppo_cfg, control_split=env_cfg.control_split)
    mem_mode = "memory" if rl.get("loading", {}).get("mode", "eager").lower() == "lazy" and rl.get("loading", {}).get("lazy_by", "count").lower().startswith("mem") else "count"
    loading_cfg = LoadingCfg(
        mode=str(rl.get("loading", {}).get("mode", "eager")),
        lazy_by=str(rl.get("loading", {}).get("lazy_by", "count")),
        batch_size=int(rl.get("loading", {}).get("batch_size", 32)),
        mem_fraction=float(rl.get("loading", {}).get("mem_fraction", 0.75)),
        iters_per_batch=int(rl.get("loading", {}).get("iters_per_batch", 1)),
        aug_per_image=int(rl.get("loading", {}).get("aug_per_image", 3)),
        aug_kinds=list(rl.get("loading", {}).get("aug_kinds", ["hflip", "vflip", "rot90"])),
        shuffle_seed=int(rl.get("loading", {}).get("shuffle_seed", 0)),
    )
    es_raw = (yaml_cfg.get("rl", {}) or {}).get("early_stop", {}) if yaml_cfg else {}
    early_cfg = EarlyStopCfg(
        enabled=bool(es_raw.get("enabled", False)),
        patience=int(es_raw.get("patience", 10)),
        min_delta_bpp=float(es_raw.get("min_delta_bpp", 0.0)),
        min_delta_db=float(es_raw.get("min_delta_db", 0.0)),
        start_after=int(es_raw.get("start_after", 0)),
    )
    stopper = EarlyStopper(early_cfg)
    adapt_raw = (yaml_cfg.get("rl", {}) or {}).get("adapt", {}) if yaml_cfg else {}
    adapt_enabled = bool(adapt_raw.get("enabled", False))
    if isinstance(adapt_raw.get("boundary_pts", None), (list, tuple)) and len(adapt_raw["boundary_pts"]) >= 2:
        (x1, y1), (x2, y2) = adapt_raw["boundary_pts"][0], adapt_raw["boundary_pts"][1]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        if abs(x2 - x1) > 1e-12:
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1
        else:
            m = 60.0; c = 11.0
    else:
        m = float(adapt_raw.get("m", 60.0))
        c = float(adapt_raw.get("c", 11.0))
    use_stat_bpp = str(adapt_raw.get("use_stats", {}).get("bpp", "q3")).lower()
    use_stat_psnr = str(adapt_raw.get("use_stats", {}).get("psnr", "q1")).lower()
    ema_beta = float(adapt_raw.get("ema_beta", 0.9))
    kb = float(adapt_raw.get("gains", {}).get("kb", 0.15))
    kp = float(adapt_raw.get("gains", {}).get("kp", 0.15))
    scale_bpp = float(adapt_raw.get("scales", {}).get("bpp", 0.10))
    scale_psnr = float(adapt_raw.get("scales", {}).get("psnr", 2.5))
    trust = float(adapt_raw.get("trust_region", 0.10))
    a_min = float(adapt_raw.get("alpha_bounds", {}).get("min", 0.2))
    a_max = float(adapt_raw.get("alpha_bounds", {}).get("max", 10.0))
    decay = float(adapt_raw.get("decay_when_ok", 0.02))
    warm_batches = int(adapt_raw.get("warmup_batches", 2))
    adapt_mode = str(adapt_raw.get("mode", "residuals")).lower()
    alpha_bpp_cur = float(alpha_bpp_base)
    alpha_psnr_cur = float(alpha_psnr_base)
    env_cfg._alpha_bpp = alpha_bpp_cur
    env_cfg._alpha_psnr = alpha_psnr_cur
    bpp_tail_ema = None
    psnr_floor_ema = None
    batches_seen = 0
    use_eager = loading_cfg.mode.lower() == "eager"
    keep_cache = bool(rl.get("loading", {}).get("keep_cache", True))
    cached_imgs: Optional[List[np.ndarray]] = None
    if use_eager and keep_cache:
        cached_imgs = preload_all_images(
            paths,
            aug_kinds=loading_cfg.aug_kinds,
            aug_per_image=loading_cfg.aug_per_image,
            seed=loading_cfg.shuffle_seed
        )
    for it in range(iters):
        epoch_vals: List[Dict[str, float]] = []
        if use_eager and keep_cache and cached_imgs is not None:
            batcher = ImageBatcher(
                paths=[],
                mode="eager",
                batch_size=loading_cfg.batch_size,
                mem_fraction=loading_cfg.mem_fraction,
                aug_per_image=loading_cfg.aug_per_image,
                aug_kinds=loading_cfg.aug_kinds,
                shuffle_seed=loading_cfg.shuffle_seed + it,
                cached_imgs=cached_imgs
            )
            num_batches = math.ceil(len(cached_imgs) / loading_cfg.batch_size)
        else:
            mem_mode2 = "memory" if loading_cfg.mode.lower() == "lazy" and loading_cfg.lazy_by.lower().startswith("mem") else "count"
            batcher = ImageBatcher(
                paths,
                mode=mem_mode2 if loading_cfg.mode.lower() == "lazy" else "count",
                batch_size=loading_cfg.batch_size,
                mem_fraction=loading_cfg.mem_fraction,
                aug_per_image=loading_cfg.aug_per_image,
                aug_kinds=loading_cfg.aug_kinds,
                shuffle_seed=loading_cfg.shuffle_seed + it
            )
            if batcher.mode == "count":
                num_batches = math.ceil(len(paths) / batcher.batch_size)
            else:
                sample_n = min(16, len(paths))
                sample_imgs = []
                try:
                    for i in range(sample_n):
                        img_i = load_image_rgb(paths[i])
                        sample_imgs.append(img_i)
                        augs_i = make_augs(img_i, loading_cfg.aug_kinds, loading_cfg.aug_per_image)
                        sample_imgs.extend(augs_i)
                    if sample_imgs:
                        est_bytes_per_unit = sum(im.nbytes for im in sample_imgs) / float(len(sample_imgs))
                    else:
                        est_bytes_per_unit = 8 * 1024 * 1024
                finally:
                    del sample_imgs
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                vm0 = psutil.virtual_memory()
                budget0 = max(128 * 1024 * 1024, int(vm0.available * loading_cfg.mem_fraction))
                est_units_per_batch0 = max(1, int(budget0 // max(1, int(est_bytes_per_unit))))
                num_batches = max(1, math.ceil(len(paths) / est_units_per_batch0))
        batch_idx = 0
        for imgs in batcher:
            batch_idx += 1
            approx_tag = "" if (use_eager and keep_cache and cached_imgs is not None) or batcher.mode == "count" else "~"
            print(f"[Epoch {it+1}/{iters}] Batch {batch_idx}/{approx_tag}{num_batches}")
            envs = build_envs(imgs, env_cfg, n_envs)
            batch_vals: List[Dict[str, float]] = []
            inner = max(1, loading_cfg.iters_per_batch)
            for _ in range(inner):
                obs, act, logp_old, adv, ret, info = trainer._gather(envs, ppo_cfg.steps_per_iter)
                vals = [i for i in info if "bpp" in i and "mse" in i]
                if vals:
                    batch_vals.extend(vals)
                    epoch_vals.extend(vals)
                trainer.update(obs, act, logp_old, adv, ret)
            if batch_vals:
                bpp_b = [v["bpp"] for v in batch_vals]
                mse_b = [v["mse"] for v in batch_vals]
                psnr_b = [10.0 * math.log10(1.0 / max(m, 1e-12)) for m in mse_b]
                stats_bpp_b = np.percentile(bpp_b, [0, 25, 50, 75, 100])
                stats_psnr_b = np.percentile(psnr_b, [0, 25, 50, 75, 100])
                avg_ssim_b = float(np.mean([v["ssim"] for v in batch_vals]))
                avg_msssim_b = float(np.mean([v["msssim"] for v in batch_vals]))
                avg_reward_b = -(WBPP * float(np.mean(bpp_b)) - WPSNR * float(np.mean(psnr_b)) - alpha_msssim * avg_msssim_b)
                print(
                    f"[Epoch {it+1}/{iters}] {batch_idx}/{approx_tag}{num_batches}  BATCH  "
                    f"bpp=[{stats_bpp_b[0]:.3f}, {stats_bpp_b[1]:.3f}, {stats_bpp_b[2]:.3f}, {stats_bpp_b[3]:.3f}, {stats_bpp_b[4]:.3f}]  "
                    f"PSNR=[{stats_psnr_b[0]:.2f}, {stats_psnr_b[1]:.2f}, {stats_psnr_b[2]:.2f}, {stats_psnr_b[3]:.2f}, {stats_psnr_b[4]:.2f}]  "
                    f"SSIM={avg_ssim_b:.3f}  MS={avg_msssim_b:.3f}  R~={avg_reward_b:.3f}"
                )
                env_cfg._bpp_cuts = (float(stats_bpp_b[1]), float(stats_bpp_b[2]), float(stats_bpp_b[3]))
                env_cfg._psnr_cuts = (float(stats_psnr_b[1]), float(stats_psnr_b[2]), float(stats_psnr_b[3]))
                env_cfg._bpp_weights = (bpp_q1, bpp_med, bpp_q3)
                env_cfg._psnr_weights = (psnr_q1, psnr_med, psnr_q3)
                print(f"[QWEIGHTS] bpp_cuts=(q1={env_cfg._bpp_cuts[0]:.3f}, med={env_cfg._bpp_cuts[1]:.3f}, q3={env_cfg._bpp_cuts[2]:.3f})  "
                      f"psnr_cuts=(q1={env_cfg._psnr_cuts[0]:.2f}, med={env_cfg._psnr_cuts[1]:.2f}, q3={env_cfg._psnr_cuts[2]:.2f})  "
                      f"wb={env_cfg._bpp_weights}  wp={env_cfg._psnr_weights}")
            if epoch_vals:
                bpp_e = [v["bpp"] for v in epoch_vals]
                mse_e = [v["mse"] for v in epoch_vals]
                psnr_e = [10.0 * math.log10(1.0 / max(m, 1e-12)) for m in mse_e]
                stats_bpp_e = np.percentile(bpp_e, [0, 25, 50, 75, 100])
                stats_psnr_e = np.percentile(psnr_e, [0, 25, 50, 75, 100])
                avg_ssim_e = float(np.mean([v["ssim"] for v in epoch_vals]))
                avg_msssim_e = float(np.mean([v["msssim"] for v in epoch_vals]))
                avg_reward_e = -(WBPP * float(np.mean(bpp_e)) - WPSNR * float(np.mean(psnr_e)) - alpha_msssim * avg_msssim_e)
                print(
                    f"[Epoch {it+1}/{iters}] {batch_idx}/{approx_tag}{num_batches}  TOTAL  "
                    f"bpp=[{stats_bpp_e[0]:.3f}, {stats_bpp_e[1]:.3f}, {stats_bpp_e[2]:.3f}, {stats_bpp_e[3]:.3f}, {stats_bpp_e[4]:.3f}]  "
                    f"PSNR=[{stats_psnr_e[0]:.2f}, {stats_psnr_e[1]:.2f}, {stats_psnr_e[2]:.2f}, {stats_psnr_e[3]:.2f}, {stats_psnr_e[4]:.2f}]  "
                    f"SSIM={avg_ssim_e:.3f}  MS={avg_msssim_e:.3f}  R~={avg_reward_e:.3f}"
                )
            if adapt_enabled and batch_vals:
                batches_seen += 1
                if use_stat_bpp == "q3":
                    bpp_tail = float(np.percentile(bpp_b, 75))
                elif use_stat_bpp == "p95":
                    bpp_tail = float(np.percentile(bpp_b, 95))
                else:
                    bpp_tail = float(np.percentile(bpp_b, 75))
                if use_stat_psnr == "q1":
                    psnr_floor = float(np.percentile(psnr_b, 25))
                elif use_stat_psnr == "median":
                    psnr_floor = float(np.percentile(psnr_b, 50))
                else:
                    psnr_floor = float(np.percentile(psnr_b, 25))
                if bpp_tail_ema is None:
                    bpp_tail_ema = bpp_tail
                else:
                    bpp_tail_ema = float(ema_beta * bpp_tail_ema + (1 - ema_beta) * bpp_tail)
                if psnr_floor_ema is None:
                    psnr_floor_ema = psnr_floor
                else:
                    psnr_floor_ema = float(ema_beta * psnr_floor_ema + (1 - ema_beta) * psnr_floor)
                if batches_seen > max(1, warm_batches):
                    psnr_req = m * bpp_tail_ema + c
                    bpp_req = (psnr_floor_ema - c) / m
                    d_psnr = max(0.0, psnr_req - psnr_floor_ema)
                    d_bpp = max(0.0, bpp_tail_ema - bpp_req)
                    if adapt_mode == "perp":
                        denom = math.sqrt(m*m + 1.0)
                        signed_d = (m * bpp_tail_ema - psnr_floor_ema + c) / denom
                        d_psnr = max(0.0, signed_d)
                        d_bpp  = max(0.0, signed_d)
                    upd_psnr = 1.0 + kp * (d_psnr / max(1e-8, scale_psnr))
                    upd_bpp  = 1.0 + kb * (d_bpp  / max(1e-8, scale_bpp))
                    upd_psnr = float(min(1.0 + trust, max(1.0 - trust, upd_psnr)))
                    upd_bpp  = float(min(1.0 + trust, max(1.0 - trust, upd_bpp)))
                    a_psnr_old = alpha_psnr_cur
                    a_bpp_old  = alpha_bpp_cur
                    if d_psnr > 0.0:
                        alpha_psnr_cur = float(np.clip(alpha_psnr_cur * upd_psnr, a_min, a_max))
                    else:
                        alpha_psnr_cur = float((1 - decay) * alpha_psnr_cur + decay * alpha_psnr_base)
                    if d_bpp > 0.0:
                        alpha_bpp_cur = float(np.clip(alpha_bpp_cur * upd_bpp, a_min, a_max))
                    else:
                        alpha_bpp_cur = float((1 - decay) * alpha_bpp_cur + decay * alpha_bpp_base)
                    env_cfg._alpha_psnr = alpha_psnr_cur
                    env_cfg._alpha_bpp  = alpha_bpp_cur
                    print(
                        f"[ADAPT] bpp_tail_ema={bpp_tail_ema:.3f} psnr_floor_ema={psnr_floor_ema:.2f} "
                        f"req_psnr={psnr_req:.2f} req_bpp={bpp_req:.3f} "
                        f"d_psnr={d_psnr:.2f} d_bpp={d_bpp:.3f} "
                        f"alpha_psnr:{a_psnr_old:.3f}->{alpha_psnr_cur:.3f} "
                        f"alpha_bpp:{a_bpp_old:.3f}->{alpha_bpp_cur:.3f}"
                    )
            del envs, imgs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if epoch_vals:
            bpp_vals = [v["bpp"] for v in epoch_vals]
            mse_vals = [v["mse"] for v in epoch_vals]
            psnr_vals = [10.0 * math.log10(1.0 / max(m, 1e-12)) for m in mse_vals]
            ssim_vals = [v["ssim"] for v in epoch_vals]
            msssim_vals = [v["msssim"] for v in epoch_vals]
            med_bpp  = float(np.median(bpp_vals))
            min_bpp  = float(np.min(bpp_vals))
            max_bpp  = float(np.max(bpp_vals))
            q1_bpp   = float(np.percentile(bpp_vals, 25))
            q3_bpp   = float(np.percentile(bpp_vals, 75))
            med_psnr = float(np.median(psnr_vals))
            min_psnr = float(np.min(psnr_vals))
            max_psnr = float(np.max(psnr_vals))
            q1_psnr  = float(np.percentile(psnr_vals, 25))
            q3_psnr  = float(np.percentile(psnr_vals, 75))
            avg_ssim = float(np.mean(ssim_vals))
            avg_msssim = float(np.mean(msssim_vals))
            msssim_med = float(np.median(msssim_vals))
            W_bpp  = bpp_q1 * q1_bpp + bpp_med * med_bpp + bpp_q3 * q3_bpp
            W_psnr = psnr_q1 * q1_psnr + psnr_med * med_psnr + psnr_q3 * q3_psnr
            W_total = alpha_bpp_cur * W_bpp - alpha_psnr_cur * W_psnr - alpha_msssim * msssim_med
            print(
                f"[RL] iter {it+1}/{iters}  "
                f"bpp_med={med_bpp:.3f} (min={min_bpp:.3f}, q1={q1_bpp:.3f}, q3={q3_bpp:.3f}, max={max_bpp:.3f})  "
                f"PSNR_med={med_psnr:.2f} dB (min={min_psnr:.2f}, q1={q1_psnr:.2f}, q3={q3_psnr:.2f}, max={max_psnr:.2f})  "
                f"SSIM={avg_ssim:.3f}  MS-SSIM={avg_msssim:.3f}  "
                f"Wbpp={W_bpp:.3f}  Wpsnr={W_psnr:.2f}  mSSIM_med={msssim_med:.3f}  Wtotal={W_total:.3f}  "
                f"alpha_bpp={alpha_bpp_cur:.3f} alpha_psnr={alpha_psnr_cur:.3f}"
            )
            should_stop, bpp_imp, db_imp = stopper.update(it, med_bpp, med_psnr)
            tag = f"(improved: bpp_med={bpp_imp}, psnr_med={db_imp}, wait={stopper.wait}/{early_cfg.patience})"
            print(f"[RL] {tag}")
            if should_stop:
                print(
                    f"[RL] early stop: no median bpp/psnr improvement for {early_cfg.patience} epochs "
                    f"(after warmup {early_cfg.start_after})."
                )
                trainer.save_torchscript(out_model)
                print(f"[RL] saved TorchScript policy -> {out_model}")
                return
        else:
            print(f"[RL] iter {it+1}/{iters}  (no metric-bearing steps this epoch)")
    trainer.save_torchscript(out_model)
    print(f"[RL] saved TorchScript policy -> {out_model}")
