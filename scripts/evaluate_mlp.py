from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.models.mlp import DeltaMLPDynamics
from src.rollout.evaluator import horizon_rmse, rollout


def valid_windows(dones: torch.Tensor, horizon: int):
    """Yield (episode, start) windows that do not cross an environment reset."""
    n, t = dones.shape
    for e in range(n):
        for start in range(0, t - horizon + 1):
            if not torch.any(dones[e, start : start + horizon]):
                yield e, start


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=Path, required=True, help="NPZ produced by collect_cartpole.py")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--max-windows", type=int, default=4096)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = DeltaMLPDynamics(ckpt["state_dim"], ckpt["action_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    d = np.load(args.episodes)
    states = torch.from_numpy(d["states"]).float()      # [E,T+1,D]
    actions = torch.from_numpy(d["actions"]).float()    # [E,T,A]
    dones = torch.from_numpy(d["dones"]).bool()         # [E,T]
    h = min(args.horizon, actions.shape[1])

    windows = list(valid_windows(dones, h))[: args.max_windows]
    if not windows:
        raise RuntimeError("No reset-free rollout windows found; reduce --horizon or collect longer episodes.")

    initial = torch.stack([states[e, s] for e, s in windows])
    action_batch = torch.stack([actions[e, s : s + h] for e, s in windows])
    target = torch.stack([states[e, s + 1 : s + h + 1] for e, s in windows])

    pred = rollout(model, initial, action_batch)
    curve = horizon_rmse(pred, target)
    print(f"evaluated_windows={len(windows)} horizon={h}")
    for i, value in enumerate(curve.tolist(), 1):
        print(f"h={i:03d} rmse={value:.6g}")


if __name__ == "__main__":
    main()
