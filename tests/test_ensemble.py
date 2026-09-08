import torch

from src.models.ensemble import ProbabilisticEnsemble


def test_ensemble_shapes():
    batch_size = 8
    state_dim = 4
    action_dim = 1
    ensemble_size = 5

    model = ProbabilisticEnsemble(
        state_dim=state_dim,
        action_dim=action_dim,
        ensemble_size=ensemble_size,
        hidden_dim=256,
    )

    state = torch.randn(
        batch_size,
        state_dim,
    )

    action = torch.randn(
        batch_size,
        action_dim,
    )

    means, logvars = model(
        state,
        action,
    )

    assert means.shape == (
        ensemble_size,
        batch_size,
        state_dim,
    )

    assert logvars.shape == (
        ensemble_size,
        batch_size,
        state_dim,
    )

    assert torch.isfinite(means).all()
    assert torch.isfinite(logvars).all()


def test_ensemble_mean_prediction():
    batch_size = 8

    model = ProbabilisticEnsemble(
        state_dim=4,
        action_dim=1,
        ensemble_size=5,
    )

    state = torch.randn(
        batch_size,
        4,
    )

    action = torch.randn(
        batch_size,
        1,
    )

    prediction = model.mean_prediction(
        state,
        action,
    )

    assert prediction.shape == (
        batch_size,
        4,
    )

    assert torch.isfinite(
        prediction
    ).all()


def test_ensemble_uncertainty_shapes():
    batch_size = 8

    model = ProbabilisticEnsemble(
        state_dim=4,
        action_dim=1,
        ensemble_size=5,
    )

    state = torch.randn(
        batch_size,
        4,
    )

    action = torch.randn(
        batch_size,
        1,
    )

    disagreement = model.disagreement(
        state,
        action,
    )

    aleatoric = model.aleatoric_variance(
        state,
        action,
    )

    assert disagreement.shape == (
        batch_size,
        4,
    )

    assert aleatoric.shape == (
        batch_size,
        4,
    )

    assert (
        disagreement >= 0
    ).all()

    assert (
        aleatoric >= 0
    ).all()


def test_ensemble_members_are_independent():
    model = ProbabilisticEnsemble(
        state_dim=4,
        action_dim=1,
        ensemble_size=5,
    )

    first_param_member_0 = next(
        model.members[0].parameters()
    )

    first_param_member_1 = next(
        model.members[1].parameters()
    )

    assert (
        first_param_member_0.data_ptr()
        != first_param_member_1.data_ptr()
    )