from __future__ import annotations

import torch
from torch import nn


class DeltaMLPDynamics(nn.Module):
    """Small deterministic next-state baseline used to validate the harness."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([state, action], dim=-1))
        return state + delta
