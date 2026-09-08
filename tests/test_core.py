import numpy as np
import torch

from src.data.dataset import TransitionArrays, load_transitions, save_transitions
from src.models.mlp import DeltaMLPDynamics
from src.rollout.evaluator import horizon_rmse, rollout


def test_dataset_roundtrip(tmp_path):
    arrays = TransitionArrays(
        state=np.zeros((5, 4), dtype=np.float32),
        action=np.ones((5, 1), dtype=np.float32),
        next_state=np.ones((5, 4), dtype=np.float32),
        done=np.array([0, 0, 0, 1, 0], dtype=bool),
    )
    path = tmp_path / "x.npz"
    save_transitions(path, arrays)
    loaded = load_transitions(path)
    assert loaded.state.shape == (5, 4)
    assert loaded.action.shape == (5, 1)
    assert loaded.done.dtype == np.bool_


def test_rollout_shapes():
    model = DeltaMLPDynamics(4, 1, hidden_dim=16)
    initial = torch.zeros(3, 4)
    actions = torch.zeros(3, 7, 1)
    pred = rollout(model, initial, actions)
    assert pred.shape == (3, 7, 4)
    rmse = horizon_rmse(pred, torch.zeros_like(pred))
    assert rmse.shape == (7,)
