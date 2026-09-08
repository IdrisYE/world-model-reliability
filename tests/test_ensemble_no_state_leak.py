import argparse
from pathlib import Path

import numpy as np
import torch

from src.data.episodes import (
    extract_episodes,
    split_episode_indices,
)

from src.models.ensemble import (
    ProbabilisticEnsemble,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--context",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--max-windows",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--complete-only",
        action="store_true",
    )

    return parser.parse_args()


def load_validation_windows(
    path,
    context,
    horizon,
    max_windows,
    seed,
    complete_only=False,
):
    data = np.load(path)

    episodes = extract_episodes(
        data["states"],
        data["actions"],
        data["dones"],
    )

    if complete_only:
        max_len = max(
            len(ep["actions"])
            for ep in episodes
        )

        episodes = [
            ep
            for ep in episodes
            if len(ep["actions"]) == max_len
        ]

    _, val_episode_indices = split_episode_indices(
        episodes,
        val_fraction=0.1,
        seed=42,
    )

    required_steps = context + horizon

    windows = []

    for ep_idx in val_episode_indices:
        episode = episodes[ep_idx]
        num_steps = len(
            episode["actions"]
        )

        if num_steps < required_steps:
            continue

        for start in range(
            0,
            num_steps - required_steps + 1,
        ):
            windows.append(
                (ep_idx, start)
            )

    rng = np.random.default_rng(seed)

    if len(windows) > max_windows:
        indices = rng.choice(
            len(windows),
            size=max_windows,
            replace=False,
        )

        windows = [
            windows[i]
            for i in indices
        ]

    state_windows = []
    action_windows = []

    for ep_idx, start in windows:
        end = start + required_steps

        state_windows.append(
            episodes[ep_idx]["states"][
                start:end
            ]
        )

        action_windows.append(
            episodes[ep_idx]["actions"][
                start:end
            ]
        )

    return (
        np.stack(state_windows).astype(
            np.float32
        ),
        np.stack(action_windows).astype(
            np.float32
        ),
    )


@torch.no_grad()
def rollout(
    model,
    states,
    actions,
    state_mean,
    state_std,
    context,
    horizon,
):
    states_norm = (
        states - state_mean
    ) / state_std

    current_state = states_norm[
        :,
        context - 1,
    ]

    predictions = []

    for h in range(horizon):

        action_index = (
            context - 1 + h
        )

        action = actions[
            :,
            action_index,
        ]

        means, _ = model(
            current_state,
            action,
        )

        predicted_delta = means.mean(
            dim=0
        )

        current_state = (
            current_state
            + predicted_delta
        )

        prediction = (
            current_state
            * state_std
            + state_mean
        )

        predictions.append(
            prediction
        )

    return torch.stack(
        predictions,
        dim=1,
    )


def main():
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model = ProbabilisticEnsemble(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
        ensemble_size=checkpoint[
            "ensemble_size"
        ],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    state_windows, action_windows = (
        load_validation_windows(
            args.episodes,
            context=args.context,
            horizon=args.horizon,
            max_windows=args.max_windows,
            seed=args.seed,
            complete_only=args.complete_only,
        )
    )

    states = torch.from_numpy(
        state_windows
    ).to(device)

    actions = torch.from_numpy(
        action_windows
    ).to(device)

    state_mean = torch.tensor(
        checkpoint["state_mean"],
        dtype=torch.float32,
        device=device,
    )

    state_std = torch.tensor(
        checkpoint["state_std"],
        dtype=torch.float32,
        device=device,
    )

    # -----------------------------
    # Normal rollout
    # -----------------------------

    pred_normal = rollout(
        model,
        states,
        actions,
        state_mean,
        state_std,
        args.context,
        args.horizon,
    )

    # -----------------------------
    # Corrupt all future GT states
    # after the context boundary.
    # -----------------------------

    corrupted_states = states.clone()

    corrupted_states[
        :,
        args.context:,
    ] = torch.randn_like(
        corrupted_states[
            :,
            args.context:,
        ]
    ) * 1000.0

    pred_corrupted = rollout(
        model,
        corrupted_states,
        actions,
        state_mean,
        state_std,
        args.context,
        args.horizon,
    )

    diff = (
        pred_normal
        - pred_corrupted
    ).abs()

    print(
        f"max prediction difference="
        f"{diff.max().item():.12f}"
    )

    print(
        f"mean prediction difference="
        f"{diff.mean().item():.12f}"
    )

    if torch.equal(
        pred_normal,
        pred_corrupted,
    ):
        print(
            "PASS: predictions are exactly "
            "identical."
        )
    else:
        print(
            "NOTE: not bitwise identical, "
            "check numerical differences above."
        )


if __name__ == "__main__":
    main()