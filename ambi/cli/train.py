# ambi/cli/train.py
from __future__ import annotations
import argparse
from pathlib import Path
import os
import yaml
import torch

def main():
    ap = argparse.ArgumentParser(description="AMBI RL training (stub PPO)")
    ap.add_argument("data", type=str, help="image file or folder for training")
    ap.add_argument("--config", type=str, default="ambi/config.yaml", help="YAML config")
    ap.add_argument("--out", type=str, default="models/ambi_policy.ts", help="output TorchScript file")
    ap.add_argument("--iters", type=int, default=5, help="PPO iterations")
    ap.add_argument("--envs", type=int, default=4, help="number of vectorized envs (batch width)")
    ap.add_argument("--workers", type=int, default=None, help="rollout stepping threads (default: = --envs)")
    ap.add_argument("--steps", type=int, default=4096, help="steps per iteration")
    ap.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="policy device")
    ap.add_argument("--torch-threads", type=int, default=1, help="torch.set_num_threads(N), default 1 to avoid oversubscription")
    ap.add_argument("--blas-threads", type=int, default=1, help="set OMP/MKL/OPENBLAS threads, default 1")
    ap.add_argument("--no-progress", action="store_true", help="disable tqdm progress bars")
    args = ap.parse_args()

    # Cap BLAS/OpenMP backends to avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", str(args.blas_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.blas_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.blas_threads))

    # Torch threads (math intra-op)
    torch.set_num_threads(args.torch_threads)

    # Honor --no-progress by setting an env var that rl.py reads
    if args.no_progress:
        os.environ["AMBI_NOPROGRESS"] = "1"

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # import here to respect torch/env thread settings
    from ambi.rl import train_rl, PPOCfg

    ppo = PPOCfg(
        lr=args.lr,
        steps_per_iter=args.steps,
        max_iters=args.iters,
        device=args.device,
        env_workers=args.workers if args.workers is not None else args.envs,
    )

    train_rl(Path(args.data), Path(args.out), cfg, ppo_cfg=ppo, n_envs=args.envs, iters=args.iters)

if __name__ == "__main__":
    main()
