import argparse
from pathlib import Path

import numpy as np
import torch

from src.data.episodes import (
    extract_episodes,
    split_episode_indices,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
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
        default=4096,
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

        print(
            f"using complete episodes only: "
            f"{len(episodes)} episodes, "
            f"length={max_len}"
        )

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

    for ep_idx, start in windows:
        end = start + required_steps

        state_windows.append(
            episodes[ep_idx]["states"][
                start:end
            ]
        )

    return np.stack(state_windows)


def main():
    args = parse_args()

    state_windows = load_validation_windows(
        args.episodes,
        context=args.context,
        horizon=args.horizon,
        max_windows=args.max_windows,
        seed=args.seed,
        complete_only=args.complete_only,
    )

    print(
        f"evaluation windows={len(state_windows)} "
        f"context={args.context} "
        f"horizon={args.horizon}"
    )

    states = torch.from_numpy(
        state_windows
    ).float()

    current_state = states[
        :,
        args.context - 1,
    ]

    squared_error_sum = torch.zeros(
        args.horizon
    )

    squared_error_dim_sum = torch.zeros(
        args.horizon,
        states.shape[-1],
    )

    num_windows = states.shape[0]
    state_dim = states.shape[-1]

    for h in range(args.horizon):

        prediction = current_state

        target = states[
            :,
            args.context + h,
        ]

        error = (
            prediction - target
        ).pow(2)

        squared_error_sum[h] = (
            error.sum()
        )

        squared_error_dim_sum[h] = (
            error.sum(dim=0)
        )

    rmse = torch.sqrt(
        squared_error_sum
        / (
            num_windows
            * state_dim
        )
    )

    rmse_per_dim = torch.sqrt(
        squared_error_dim_sum
        / num_windows
    )

    print("\nPersistence rollout RMSE:")

    report_horizons = [
        1,
        2,
        5,
        10,
        20,
        25,
        30,
        40,
        50,
    ]

    for h in report_horizons:
        if h <= args.horizon:
            print(
                f"h={h:02d} "
                f"rmse={rmse[h - 1]:.6f}"
            )

    print(
        "\nPersistence per-dimension "
        "RMSE:"
    )

    for h in report_horizons:
        if h <= args.horizon:

            values = rmse_per_dim[h - 1]

            dim_text = " ".join(
                f"dim{i}={value:.6f}"
                for i, value
                in enumerate(values)
            )

            print(
                f"h={h:02d} "
                f"{dim_text}"
            )


if __name__ == "__main__":
    main()