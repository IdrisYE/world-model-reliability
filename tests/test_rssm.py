import torch

from src.models.rssm import DreamerV3RSSM


def test_rssm_shapes():
    batch_size = 8
    state_dim = 4
    action_dim = 1

    model = DreamerV3RSSM(
        state_dim=state_dim,
        action_dim=action_dim,
        deter_dim=256,
        hidden_dim=256,
        stoch_groups=32,
        classes=16,
        embed_dim=128,
    )

    prev_state = model.initial(batch_size)

    observation = torch.randn(batch_size, state_dim)
    action = torch.randn(batch_size, action_dim)

    state, prior_logits, posterior_logits = model.observe_step(
        prev_state,
        action,
        observation,
    )

    assert state.deter.shape == (batch_size, 256)
    assert state.stoch.shape == (batch_size, 32 * 16)

    assert prior_logits.shape == (batch_size, 32, 16)
    assert posterior_logits.shape == (batch_size, 32, 16)

    prediction = model.predict_state(state)

    assert prediction.shape == (batch_size, state_dim)

    dyn_loss, rep_loss = model.kl_losses(
        prior_logits,
        posterior_logits,
    )

    assert dyn_loss.ndim == 0
    assert rep_loss.ndim == 0
    assert torch.isfinite(dyn_loss)
    assert torch.isfinite(rep_loss)


def test_rssm_imagination_step():
    batch_size = 8

    model = DreamerV3RSSM(
        state_dim=4,
        action_dim=1,
    )

    state = model.initial(batch_size)
    action = torch.randn(batch_size, 1)

    next_state, prior_logits = model.imagine_step(
        state,
        action,
    )

    assert next_state.deter.shape == (batch_size, 256)
    assert next_state.stoch.shape == (batch_size, 32 * 16)
    assert prior_logits.shape == (batch_size, 32, 16)

    prediction = model.predict_state(next_state)

    assert prediction.shape == (batch_size, 4)