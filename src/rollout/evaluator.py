from __future__ import annotations

import torch


@torch.no_grad()
def rollout(model, initial_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Autoregressive rollout.

    Args:
        initial_state: [B, state_dim]
        actions: [B, H, action_dim]
    Returns:
        predicted states: [B, H, state_dim]
    """
    state = initial_state
    predictions = []
    for t in range(actions.shape[1]):
        state = model(state, actions[:, t])
        predictions.append(state)
    return torch.stack(predictions, dim=1)


def horizon_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RMSE at each rollout step; returns [H]."""
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=(0, 2)))
