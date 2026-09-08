from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DreamerV3RSSM(nn.Module):
    """
    Simplified DreamerV3-style RSSM for low-dimensional state prediction.

    Keeps:
      - deterministic recurrent state
      - discrete categorical stochastic state
      - prior and posterior
      - unimix categorical distributions
      - straight-through samples
      - dynamics + representation KL losses

    Removes:
      - actor / critic
      - reward prediction
      - continuation prediction
      - image encoder / decoder
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        deter_dim: int = 256,
        hidden_dim: int = 256,
        stoch_groups: int = 32,
        classes: int = 16,
        embed_dim: int = 128,
        unimix: float = 0.01,
        free_nats: float = 1.0,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_groups = stoch_groups
        self.classes = classes
        self.stoch_dim = stoch_groups * classes
        self.unimix = unimix
        self.free_nats = free_nats

        # Current observation -> embedding.
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.RMSNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.RMSNorm(embed_dim),
            nn.SiLU(),
        )

        # Previous stochastic state + action -> GRU input.
        self.recurrent_input = nn.Sequential(
            nn.Linear(self.stoch_dim + action_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(),
        )

        self.gru = nn.GRUCell(hidden_dim, deter_dim)

        # Dynamics prior p(z_t | h_t)
        self.prior_net = MLP(
            deter_dim,
            hidden_dim,
            self.stoch_dim,
        )

        # Posterior q(z_t | h_t, observation_t)
        self.posterior_net = MLP(
            deter_dim + embed_dim,
            hidden_dim,
            self.stoch_dim,
        )

        # Decode latent feature back to simulator state.
        self.state_head = nn.Sequential(
            nn.Linear(deter_dim + self.stoch_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def initial(self, batch_size: int, device=None) -> RSSMState:
        device = device or next(self.parameters()).device

        deter = torch.zeros(
            batch_size,
            self.deter_dim,
            device=device,
        )

        # Uniform categorical initial stochastic state.
        stoch = torch.zeros(
            batch_size,
            self.stoch_groups,
            self.classes,
            device=device,
        )
        stoch[..., 0] = 1.0

        return RSSMState(
            deter=deter,
            stoch=stoch.reshape(batch_size, -1),
        )

    def _reshape_logits(self, logits):
        return logits.view(
            logits.shape[0],
            self.stoch_groups,
            self.classes,
        )

    def _unimix_probs(self, logits):
        probs = F.softmax(logits, dim=-1)

        if self.unimix > 0:
            probs = (
                (1.0 - self.unimix) * probs
                + self.unimix / self.classes
            )

        return probs

    def _sample_straight_through(self, logits):
        probs = self._unimix_probs(logits)

        indices = torch.distributions.Categorical(
            probs=probs
        ).sample()

        sample = F.one_hot(
            indices,
            num_classes=self.classes,
        ).float()

        # Straight-through estimator:
        # forward = discrete one-hot
        # backward = gradients through probabilities
        sample = sample + probs - probs.detach()

        return sample

    def _prior(self, deter):
        logits = self.prior_net(deter)
        return self._reshape_logits(logits)

    def _posterior(self, deter, embed):
        x = torch.cat([deter, embed], dim=-1)
        logits = self.posterior_net(x)
        return self._reshape_logits(logits)

    def observe_step(
        self,
        prev_state: RSSMState,
        prev_action: torch.Tensor,
        observation: torch.Tensor,
    ):
        """
        Real observation available.

        prev latent + action
            -> deterministic transition
            -> prior
            -> posterior corrected by observation
        """

        recurrent_input = torch.cat(
            [prev_state.stoch, prev_action],
            dim=-1,
        )

        recurrent_input = self.recurrent_input(recurrent_input)

        deter = self.gru(
            recurrent_input,
            prev_state.deter,
        )

        prior_logits = self._prior(deter)

        embed = self.encoder(observation)
        posterior_logits = self._posterior(deter, embed)

        posterior_sample = self._sample_straight_through(
            posterior_logits
        )

        stoch = posterior_sample.reshape(
            observation.shape[0],
            -1,
        )

        state = RSSMState(
            deter=deter,
            stoch=stoch,
        )

        return state, prior_logits, posterior_logits

    def imagine_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
        deterministic: bool = False,
    ):
        recurrent_input = torch.cat(
            [prev_state.stoch, action],
            dim=-1,
        )

        recurrent_input = self.recurrent_input(recurrent_input)

        deter = self.gru(
            recurrent_input,
            prev_state.deter,
        )

        prior_logits = self._prior(deter)

        if deterministic:
            prior_sample = self._mode(prior_logits)
        else:
            prior_sample = self._sample_straight_through(
                prior_logits
            )

        stoch = prior_sample.reshape(
            action.shape[0],
            -1,
        )

        state = RSSMState(
            deter=deter,
            stoch=stoch,
        )

        return state, prior_logits

    def features(self, state: RSSMState):
        return torch.cat(
            [state.deter, state.stoch],
            dim=-1,
        )

    def predict_state(self, state: RSSMState):
        return self.state_head(
            self.features(state)
        )

    def kl_losses(
        self,
        prior_logits,
        posterior_logits,
    ):
        """
        DreamerV3-style asymmetric KL losses.

        Dynamics loss:
            train prior toward stop-gradient posterior

        Representation loss:
            train posterior toward stop-gradient prior
        """

        prior_probs = self._unimix_probs(prior_logits)
        posterior_probs = self._unimix_probs(posterior_logits)

        prior_log_probs = torch.log(prior_probs + 1e-8)
        posterior_log_probs = torch.log(posterior_probs + 1e-8)

        # KL(stop_grad(q) || p)
        dyn_kl = torch.sum(
            posterior_probs.detach()
            * (
                posterior_log_probs.detach()
                - prior_log_probs
            ),
            dim=-1,
        )

        # KL(q || stop_grad(p))
        rep_kl = torch.sum(
            posterior_probs
            * (
                posterior_log_probs
                - prior_log_probs.detach()
            ),
            dim=-1,
        )

        # Sum categorical groups, then average batch.
        dyn_kl = dyn_kl.sum(dim=-1)
        rep_kl = rep_kl.sum(dim=-1)

        free = torch.tensor(
            self.free_nats,
            device=dyn_kl.device,
        )

        dyn_loss = torch.maximum(dyn_kl, free).mean()
        rep_loss = torch.maximum(rep_kl, free).mean()

        return dyn_loss, rep_loss
    
    def _mode(self, logits):
        indices = torch.argmax(logits, dim=-1)

        sample = F.one_hot(
            indices,
            num_classes=self.classes,
        ).float()

        return sample