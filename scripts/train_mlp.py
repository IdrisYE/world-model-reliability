from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.data.dataset import TransitionDataset
from src.models.mlp import DeltaMLPDynamics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("checkpoints/mlp.pt"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    ds = TransitionDataset(args.data)
    n_val = max(1, int(0.1 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    state_dim = ds.state.shape[-1]
    action_dim = ds.action.shape[-1]
    model = DeltaMLPDynamics(state_dim, action_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for s, a, ns in train_loader:
            pred = model(s, a)
            loss = torch.mean((pred - ns) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * s.shape[0]
            count += s.shape[0]

        model.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for s, a, ns in val_loader:
                val_total += torch.mean((model(s, a) - ns) ** 2).item() * s.shape[0]
                val_count += s.shape[0]
        print(f"epoch={epoch+1:03d} train_mse={total/count:.6g} val_mse={val_total/val_count:.6g}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "state_dim": state_dim,
        "action_dim": action_dim,
    }, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
