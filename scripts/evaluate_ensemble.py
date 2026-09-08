import argparse
from pathlib import Path
import csv

import numpy as np
import torch

from src.data.episodes import extract_episodes, split_episode_indices
from src.models.ensemble import ProbabilisticEnsemble


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
        default=Path("results/ensemble_rollout_metrics.csv"),
    )

    parser.add_argument(
        "--complete-only",
        action="store_true",
    )


    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--max-windows", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)

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

        states = episode["states"]
        actions = episode["actions"]

        num_steps = len(actions)

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
        episode = episodes[ep_idx]

        end = start + required_steps

        state_windows.append(
            episode["states"][start:end]
        )

        action_windows.append(
            episode["actions"][start:end]
        )

    return (
        np.stack(state_windows).astype(np.float32),
        np.stack(action_windows).astype(np.float32),
    )


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"device={device}")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    state_dim = checkpoint["state_dim"]
    action_dim = checkpoint["action_dim"]
    ensemble_size = checkpoint["ensemble_size"]
    hidden_dim = checkpoint["hidden_dim"]

    state_mean = np.asarray(
        checkpoint["state_mean"],
        dtype=np.float32,
    )

    state_std = np.asarray(
        checkpoint["state_std"],
        dtype=np.float32,
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

    print(
        f"evaluation windows={len(state_windows)} "
        f"context={args.context} "
        f"horizon={args.horizon}"
    )

    states = torch.from_numpy(
        state_windows
    ).to(device)

    actions = torch.from_numpy(
        action_windows
    ).to(device)

    state_mean_t = torch.tensor(
        state_mean,
        dtype=torch.float32,
        device=device,
    )

    state_std_t = torch.tensor(
        state_std,
        dtype=torch.float32,
        device=device,
    )

    states_norm = (
        states - state_mean_t
    ) / state_std_t

    # -------------------------------------------------
    # Teacher-forced one-step prediction
    # -------------------------------------------------

    with torch.no_grad():

        current_state = (states_norm[:, args.context - 1])

        action = actions[:, args.context - 1]

        means, _ = model(current_state, action)

        predicted_delta = means.mean(dim=0)

        predicted_next_norm = (
            current_state
            + predicted_delta
        )

        predicted_next = (
            predicted_next_norm
            * state_std_t
            + state_mean_t
        )

        target_next = states[
            :,
            args.context,
        ]

        error = (
            predicted_next
            - target_next
        )

        one_step_rmse = torch.sqrt(
            torch.mean(
                error.pow(2)
            )
        )

        one_step_per_dim = torch.sqrt(
            torch.mean(
                error.pow(2),
                dim=0,
            )
        )

    print(
        "teacher-forced one-step prior RMSE="
        f"{one_step_rmse.item():.6f}"
    )

    print(
        "per-dimension RMSE="
        f"{one_step_per_dim.cpu().numpy()}"
    )

    # -------------------------------------------------
    # Autonomous rollout
    # -------------------------------------------------

    rollout_predictions = []

    with torch.no_grad():

        # Start rollout at the end of context.
        current_state = (
            states_norm[:, args.context - 1]
        )

        for h in range(args.horizon):

            action_index = (
                args.context - 1 + h
            )

            action = actions[
                :,
                action_index,
            ]

            means, _ = model(
                current_state,
                action,
            )

            # Mean over ensemble members.
            predicted_delta = means.mean(
                dim=0
            )

            current_state = (
                current_state
                + predicted_delta
            )

            prediction = (
                current_state
                * state_std_t
                + state_mean_t
            )

            rollout_predictions.append(
                prediction
            )

        rollout_predictions = torch.stack(
            rollout_predictions,
            dim=1,
        )

    rollout_targets = states[
        :,
        args.context:
        args.context + args.horizon,
    ]

    rollout_errors = (
        rollout_predictions
        - rollout_targets
    )

    rmse_by_horizon = torch.sqrt(
        torch.mean(
            rollout_errors.pow(2),
            dim=(0, 2),
        )
    )

    per_dim_rmse = torch.sqrt(
        torch.mean(
            rollout_errors.pow(2),
            dim=0,
        )
    )

    # -------------------------------------------------
    # Save CSV
    # -------------------------------------------------

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        args.out,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        dim_headers = [
            f"dim{i}_rmse"
            for i in range(state_dim)
        ]

        writer.writerow(
            [
                "metric",
                "horizon",
                "rmse",
                *dim_headers,
            ]
        )
        # Teacher-forced one-step metric.
        writer.writerow(
            [
                "teacher_forced_one_step",
                1,
                one_step_rmse.item(),
                *one_step_per_dim.cpu().numpy(),
            ]
        )

        # Autonomous rollout metrics.
        for h in range(args.horizon):

            writer.writerow(
                [
                    "rollout",
                    h + 1,
                    rmse_by_horizon[h].item(),
                    *per_dim_rmse[h].cpu().numpy(),
                ]
            )

    print()
    print(
        f"saved rollout metrics to {args.out}"
    )

    # -------------------------------------------------
    # Print selected horizons
    # -------------------------------------------------

    selected = [
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

    print()
    print("Rollout RMSE:")

    for h in selected:

        if h <= args.horizon:
            print(
                f"h={h:02d} "
                f"rmse="
                f"{rmse_by_horizon[h - 1].item():.6f}"
            )

    print()
    print("Per-dimension rollout RMSE:")

    for h in selected:

        if h <= args.horizon:

            values = (
                per_dim_rmse[
                    h - 1
                ]
                .cpu()
                .numpy()
            )

            dim_text = " ".join(
                f"dim{i}={value:.6f}"
                for i, value in enumerate(values)
            )

            print(
                f"h={h:02d} "
                f"{dim_text}"
            )


if __name__ == "__main__":
    main()