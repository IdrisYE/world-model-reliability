from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F



@dataclass
class TSSMState:
    stoch: torch.Tensor
    history: torch.Tensor


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TSSM(nn.Module):
    """
    Small TSSM-style Transformer world model for direct-state dynamics.

    Core components:
        - state encoder
        - action encoder
        - causal Transformer history model
        - discrete stochastic latent
        - prior and posterior
        - delta-state prediction head
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        model_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 256,
        stoch_groups: int = 16,
        classes: int = 16,
        hidden_dim: int = 256,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        unimix: float = 0.01,
        free_nats: float = 1.0,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_dim = model_dim

        self.stoch_groups = stoch_groups
        self.classes = classes
        self.stoch_dim = stoch_groups * classes

        self.max_seq_len = max_seq_len
        self.unimix = unimix
        self.free_nats = free_nats

        # -------------------------------------------------
        # Input encoders
        # -------------------------------------------------

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, model_dim),
            nn.RMSNorm(model_dim),
            nn.SiLU(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, model_dim),
            nn.RMSNorm(model_dim),
            nn.SiLU(),
        )

        # Previous stochastic latent also contributes
        # to the dynamics token.
        self.stoch_encoder = nn.Sequential(
            nn.Linear(self.stoch_dim, model_dim),
            nn.RMSNorm(model_dim),
            nn.SiLU(),
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                max_seq_len,
                model_dim,
            )
        )

        # -------------------------------------------------
        # Transformer backbone
        # -------------------------------------------------

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

        # -------------------------------------------------
        # Stochastic latent
        # -------------------------------------------------

        # p(z_t | transformer history)
        self.prior_net = MLP(
            model_dim,
            hidden_dim,
            self.stoch_dim,
        )

        # q(z_t | transformer history, observation_t)
        self.posterior_net = MLP(
            model_dim + model_dim,
            hidden_dim,
            self.stoch_dim,
        )

        # -------------------------------------------------
        # Delta-state decoder
        # -------------------------------------------------

        self.delta_head = nn.Sequential(
            nn.Linear(
                model_dim + self.stoch_dim,
                hidden_dim,
            ),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(
                hidden_dim,
                state_dim,
            ),
        )

    # -----------------------------------------------------
    # Distribution helpers
    # -----------------------------------------------------

    def _reshape_logits(self, logits):
        return logits.view(
            logits.shape[0],
            self.stoch_groups,
            self.classes,
        )

    def _unimix_probs(self, logits):
        probs = F.softmax(
            logits,
            dim=-1,
        )

        if self.unimix > 0:
            probs = (
                (1.0 - self.unimix) * probs
                + self.unimix / self.classes
            )

        return probs

    def _sample_straight_through(
        self,
        logits,
    ):
        probs = self._unimix_probs(
            logits
        )

        indices = torch.distributions.Categorical(
            probs=probs
        ).sample()

        sample = F.one_hot(
            indices,
            num_classes=self.classes,
        ).float()

        sample = (
            sample
            + probs
            - probs.detach()
        )

        return sample

    def _mode(self, logits):
        indices = torch.argmax(
            logits,
            dim=-1,
        )

        return F.one_hot(
            indices,
            num_classes=self.classes,
        ).float()

    # -----------------------------------------------------
    # Causal Transformer
    # -----------------------------------------------------

    def _causal_mask(
        self,
        length,
        device,
    ):
        return torch.triu(
            torch.full(
                (length, length),
                float("-inf"),
                device=device,
            ),
            diagonal=1,
        )

    def _run_transformer(
        self,
        history,
    ):
        """
        history:
            [B, T, model_dim]

        returns:
            [B, T, model_dim]
        """

        length = history.shape[1]

        if length > self.max_seq_len:
            history = history[
                :,
                -self.max_seq_len:,
            ]

            length = self.max_seq_len

        x = (
            history
            + self.position_embedding[
                :,
                :length,
            ]
        )

        mask = self._causal_mask(
            length,
            x.device,
        )

        return self.transformer(
            x,
            mask=mask,
        )

    # -----------------------------------------------------
    # Initial state
    # -----------------------------------------------------

    def initial(
        self,
        batch_size,
        device=None,
    ):
        device = (
            device
            or next(self.parameters()).device
        )

        stoch = torch.zeros(
            batch_size,
            self.stoch_groups,
            self.classes,
            device=device,
        )

        stoch[..., 0] = 1.0

        stoch = stoch.reshape(
            batch_size,
            -1,
        )

        history = torch.empty(
            batch_size,
            0,
            self.model_dim,
            device=device,
        )

        return TSSMState(
            stoch=stoch,
            history=history,
        )

    # -----------------------------------------------------
    # Observation step
    # -----------------------------------------------------

    def observe_step(
        self,
        prev_state: TSSMState,
        prev_action: torch.Tensor,
        observation: torch.Tensor,
    ):
        """
        Observation available.

        Previous latent + previous action are converted into
        a dynamics token and appended to Transformer history.

        The real observation is used only by the posterior.
        """

        action_embed = self.action_encoder(
            prev_action
        )

        stoch_embed = self.stoch_encoder(
            prev_state.stoch
        )

        dynamics_token = (
            action_embed
            + stoch_embed
        ).unsqueeze(1)

        history = torch.cat(
            [
                prev_state.history,
                dynamics_token,
            ],
            dim=1,
        )

        transformer_output = (
            self._run_transformer(
                history
            )
        )

        deter = transformer_output[:, -1]

        # Prior
        prior_logits = self._reshape_logits(
            self.prior_net(deter)
        )

        # Posterior uses real observation.
        obs_embed = self.state_encoder(
            observation
        )

        posterior_input = torch.cat(
            [
                deter,
                obs_embed,
            ],
            dim=-1,
        )

        posterior_logits = (
            self._reshape_logits(
                self.posterior_net(
                    posterior_input
                )
            )
        )

        posterior_sample = (
            self._sample_straight_through(
                posterior_logits
            )
        )

        stoch = posterior_sample.reshape(
            observation.shape[0],
            -1,
        )

        state = TSSMState(
            stoch=stoch,
            history=history,
        )

        return (
            state,
            deter,
            prior_logits,
            posterior_logits,
        )

    # -----------------------------------------------------
    # Imagination step
    # -----------------------------------------------------

    def imagine_step(
        self,
        prev_state: TSSMState,
        action: torch.Tensor,
        deterministic: bool = False,
    ):
        """
        Prior-only world-model step.
        """

        action_embed = self.action_encoder(
            action
        )

        stoch_embed = self.stoch_encoder(
            prev_state.stoch
        )

        dynamics_token = (
            action_embed
            + stoch_embed
        ).unsqueeze(1)

        history = torch.cat(
            [
                prev_state.history,
                dynamics_token,
            ],
            dim=1,
        )

        transformer_output = (
            self._run_transformer(
                history
            )
        )

        deter = transformer_output[:, -1]

        prior_logits = self._reshape_logits(
            self.prior_net(deter)
        )

        if deterministic:
            prior_sample = self._mode(
                prior_logits
            )
        else:
            prior_sample = (
                self._sample_straight_through(
                    prior_logits
                )
            )

        stoch = prior_sample.reshape(
            action.shape[0],
            -1,
        )

        state = TSSMState(
            stoch=stoch,
            history=history,
        )

        return (
            state,
            deter,
            prior_logits,
        )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    def predict_delta(
        self,
        deter,
        state: TSSMState,
    ):
        features = torch.cat(
            [
                deter,
                state.stoch,
            ],
            dim=-1,
        )

        return self.delta_head(
            features
        )

    # -----------------------------------------------------
    # KL
    # -----------------------------------------------------

    def kl_losses(
        self,
        prior_logits,
        posterior_logits,
    ):
        prior_probs = self._unimix_probs(
            prior_logits
        )

        posterior_probs = (
            self._unimix_probs(
                posterior_logits
            )
        )

        prior_log_probs = torch.log(
            prior_probs + 1e-8
        )

        posterior_log_probs = torch.log(
            posterior_probs + 1e-8
        )

        dyn_kl = torch.sum(
            posterior_probs.detach()
            * (
                posterior_log_probs.detach()
                - prior_log_probs
            ),
            dim=-1,
        )

        rep_kl = torch.sum(
            posterior_probs
            * (
                posterior_log_probs
                - prior_log_probs.detach()
            ),
            dim=-1,
        )

        dyn_kl = dyn_kl.sum(dim=-1)
        rep_kl = rep_kl.sum(dim=-1)

        free = torch.tensor(
            self.free_nats,
            device=dyn_kl.device,
        )

        dyn_loss = torch.maximum(
            dyn_kl,
            free,
        ).mean()

        rep_loss = torch.maximum(
            rep_kl,
            free,
        ).mean()

        return dyn_loss, rep_loss