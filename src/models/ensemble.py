import torch
import torch.nn as nn


class ProbabilisticDynamicsMember(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        input_dim = state_dim + action_dim
        output_dim = 2 * state_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.state_dim = state_dim

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        x = torch.cat(
            [state, action],
            dim=-1,
        )

        out = self.net(x)

        mean, logvar = torch.split(
            out,
            self.state_dim,
            dim=-1,
        )

        # Keep variance numerically reasonable.
        logvar = torch.clamp(
            logvar,
            min=-10.0,
            max=2.0,
        )

        return mean, logvar


class ProbabilisticEnsemble(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size

        self.members = nn.ModuleList(
            [
                ProbabilisticDynamicsMember(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                )
                for _ in range(ensemble_size)
            ]
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        means = []
        logvars = []

        for member in self.members:
            mean, logvar = member(
                state,
                action,
            )

            means.append(mean)
            logvars.append(logvar)

        means = torch.stack(
            means,
            dim=0,
        )

        logvars = torch.stack(
            logvars,
            dim=0,
        )

        return means, logvars

    @torch.no_grad()
    def mean_prediction(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        means, logvars = self(
            state,
            action,
        )

        ensemble_mean = means.mean(
            dim=0
        )

        return ensemble_mean

    @torch.no_grad()
    def disagreement(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        means, _ = self(
            state,
            action,
        )

        # Epistemic uncertainty:
        # variance across ensemble means.
        return means.var(
            dim=0,
            unbiased=False,
        )

    @torch.no_grad()
    def aleatoric_variance(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        _, logvars = self(
            state,
            action,
        )

        variances = torch.exp(
            logvars
        )

        return variances.mean(
            dim=0
        )