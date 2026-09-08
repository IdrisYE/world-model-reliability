import argparse
import csv
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
        "--out",
        type=Path,
        required=True,
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


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"device={device}")

    # -------------------------------------------------
    # Load episodes
    # -------------------------------------------------

    data = np.load(args.episodes)

    episodes = extract_episodes(
        data["states"],
        data["actions"],
        data["dones"],
    )

    if args.complete_only:
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

    _, val_indices = split_episode_indices(
        episodes,
        val_fraction=0.1,
        seed=42,
    )

    print(
        f"validation episodes="
        f"{len(val_indices)}"
    )

    # -------------------------------------------------
    # Load model
    # -------------------------------------------------

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    state_dim = checkpoint["state_dim"]
    action_dim = checkpoint["action_dim"]
    ensemble_size = checkpoint["ensemble_size"]
    hidden_dim = checkpoint["hidden_dim"]

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

    model = ProbabilisticEnsemble(
        state_dim=state_dim,
        action_dim=action_dim,
        ensemble_size=ensemble_size,
        hidden_dim=hidden_dim,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # -------------------------------------------------
    # CSV
    # -------------------------------------------------

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dim_headers = [
        f"dim{i}_residual"
        for i in range(state_dim)
    ]

    row_count = 0

    with open(
        args.out,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "model",
                "episode",
                "t",
                "prior_residual",
                "prior_position_residual",
                "prior_velocity_residual",
                *dim_headers,
            ]
        )

        # ---------------------------------------------
        # Every transition in every validation episode
        # ---------------------------------------------

        with torch.no_grad():

            for ep_idx in val_indices:

                episode = episodes[ep_idx]

                states = torch.from_numpy(
                    episode["states"]
                ).float().to(device)

                actions = torch.from_numpy(
                    episode["actions"]
                ).float().to(device)

                states_norm = (
                    states - state_mean
                ) / state_std

                num_steps = len(actions)

                for t in range(num_steps):

                    current_state = (
                        states_norm[t]
                        .unsqueeze(0)
                    )

                    action = (
                        actions[t]
                        .unsqueeze(0)
                    )

                    means, _ = model(
                        current_state,
                        action,
                    )

                    # Mean prediction across ensemble members.
                    predicted_delta = (
                        means.mean(dim=0)
                    )

                    predicted_next_norm = (
                        current_state
                        + predicted_delta
                    )

                    predicted_next = (
                        predicted_next_norm
                        * state_std
                        + state_mean
                    )

                    target = (
                        states[t + 1]
                        .unsqueeze(0)
                    )

                    error = (
                        predicted_next
                        - target
                    ).squeeze(0)

                    squared_error = (
                        error.pow(2)
                    )

                    prior_residual = torch.sqrt(
                        squared_error.mean()
                    )

                    position_residual = torch.sqrt(
                        squared_error[:9].mean()
                    )

                    velocity_residual = torch.sqrt(
                        squared_error[9:18].mean()
                    )

                    dim_residuals = (
                        error.abs()
                    )

                    writer.writerow(
                        [
                            "ensemble",
                            int(ep_idx),
                            t,
                            float(
                                prior_residual.item()
                            ),
                            float(
                                position_residual.item()
                            ),
                            float(
                                velocity_residual.item()
                            ),
                            *[
                                float(x)
                                for x in (
                                    dim_residuals
                                    .cpu()
                                    .numpy()
                                )
                            ],
                        ]
                    )

                    row_count += 1

    print(
        f"saved {row_count} residuals "
        f"to {args.out}"
    )


if __name__ == "__main__":
    main()