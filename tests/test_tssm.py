import torch

from src.models.tssm import TSSM


def test_tssm_observe_shapes():
    batch_size = 8
    state_dim = 4
    action_dim = 1

    model = TSSM(
        state_dim=state_dim,
        action_dim=action_dim,
        model_dim=128,
        num_layers=4,
        num_heads=4,
        ff_dim=256,
        stoch_groups=16,
        classes=16,
        hidden_dim=256,
        max_seq_len=128,
    )

    state = model.initial(batch_size)

    observation = torch.randn(
        batch_size,
        state_dim,
    )

    action = torch.randn(
        batch_size,
        action_dim,
    )

    (
        next_state,
        deter,
        prior_logits,
        posterior_logits,
    ) = model.observe_step(
        state,
        action,
        observation,
    )

    assert deter.shape == (
        batch_size,
        128,
    )

    assert next_state.stoch.shape == (
        batch_size,
        16 * 16,
    )

    assert next_state.history.shape == (
        batch_size,
        1,
        128,
    )

    assert prior_logits.shape == (
        batch_size,
        16,
        16,
    )

    assert posterior_logits.shape == (
        batch_size,
        16,
        16,
    )

    delta = model.predict_delta(
        deter,
        next_state,
    )

    assert delta.shape == (
        batch_size,
        state_dim,
    )


def test_tssm_imagine_shapes():
    batch_size = 8

    model = TSSM(
        state_dim=4,
        action_dim=1,
    )

    state = model.initial(batch_size)

    action = torch.randn(
        batch_size,
        1,
    )

    (
        next_state,
        deter,
        prior_logits,
    ) = model.imagine_step(
        state,
        action,
    )

    assert deter.shape == (
        batch_size,
        128,
    )

    assert next_state.stoch.shape == (
        batch_size,
        16 * 16,
    )

    assert next_state.history.shape == (
        batch_size,
        1,
        128,
    )

    assert prior_logits.shape == (
        batch_size,
        16,
        16,
    )

    delta = model.predict_delta(
        deter,
        next_state,
    )

    assert delta.shape == (
        batch_size,
        4,
    )


def test_tssm_history_grows():
    batch_size = 4

    model = TSSM(
        state_dim=4,
        action_dim=1,
    )

    state = model.initial(batch_size)

    observation = torch.randn(
        batch_size,
        4,
    )

    action = torch.randn(
        batch_size,
        1,
    )

    for step in range(5):

        (
            state,
            deter,
            prior_logits,
            posterior_logits,
        ) = model.observe_step(
            state,
            action,
            observation,
        )

        assert state.history.shape[1] == (
            step + 1
        )