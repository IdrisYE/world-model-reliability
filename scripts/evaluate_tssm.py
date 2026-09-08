import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from src.models.tssm import TSSM
from src.data.episodes import extract_episodes, split_episode_indices

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
        "--horizon",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--context",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-windows",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path(
            "results/tssm_rollout_metrics.csv"
        ),
    )

    parser.add_argument(
        "--complete-only",
        action="store_true",
    )

    parser.add_argument(
        "--deterministic",
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
            episode["states"][
                start:end
            ]
        )

        action_windows.append(
            episode["actions"][
                start:end
            ]
        )

    return (
        np.stack(state_windows),
        np.stack(action_windows),
    )

@torch.no_grad()
def evaluate_one_step_prior(
    model,
    state_windows,
    action_windows,
    context,
    batch_size,
    device,
    state_mean,
    state_std,
    deterministic=False,
):
    model.eval()

    squared_error_sum = torch.zeros(
        model.state_dim,
        device=device,
    )

    sample_count = 0

    num_windows = (
        state_windows.shape[0]
    )

    for batch_start in range(
        0,
        num_windows,
        batch_size,
    ):

        batch_end = min(
            batch_start + batch_size,
            num_windows,
        )

        raw_states = torch.from_numpy(
            state_windows[
                batch_start:batch_end
            ]
        ).to(device)

        actions = torch.from_numpy(
            action_windows[
                batch_start:batch_end
            ]
        ).to(device)

        states = (
            raw_states
            - state_mean
        ) / state_std

        batch = states.shape[0]

        tssm_state = model.initial(
            batch,
            device=device,
        )

        prev_action = torch.zeros(
            batch,
            model.action_dim,
            device=device,
        )

        deter = None

        # -----------------------------
        # Posterior context warm-up
        # -----------------------------

        for t in range(context):

            (
                tssm_state,
                deter,
                _,
                _,
            ) = model.observe_step(
                tssm_state,
                prev_action,
                states[:, t],
            )

            prev_action = (
                actions[:, t]
            )

        current_state = (
            states[:, context - 1]
        )

        # -----------------------------
        # One prior transition
        # -----------------------------

        (
            prior_state,
            prior_deter,
            _,
        ) = model.imagine_step(
            tssm_state,
            prev_action,
            deterministic=deterministic,
        )

        predicted_delta = (
            model.predict_delta(
                prior_deter,
                prior_state,
            )
        )

        predicted_next_norm = (
            current_state
            + predicted_delta
        )

        prediction = (
            predicted_next_norm
            * state_std
            + state_mean
        )

        target = (
            raw_states[:, context]
        )

        error = (
            prediction - target
        ).pow(2)

        squared_error_sum += (
            error.sum(dim=0)
        )

        sample_count += batch

    mse_per_dim = (
        squared_error_sum
        / sample_count
    )

    rmse_per_dim = torch.sqrt(
        mse_per_dim
    )

    overall_rmse = torch.sqrt(
        mse_per_dim.mean()
    )

    return (
        overall_rmse.item(),
        rmse_per_dim.cpu().numpy(),
    )


@torch.no_grad()
def evaluate(
    model,
    state_windows,
    action_windows,
    context,
    horizon,
    batch_size,
    device,
    state_mean,
    state_std,
    deterministic=False,
):
    model.eval()

    squared_error_sum = torch.zeros(
        horizon,
        device=device,
    )

    squared_error_dim_sum = (
        torch.zeros(
            horizon,
            model.state_dim,
            device=device,
        )
    )

    element_count = 0
    sample_count = 0

    num_windows = (
        state_windows.shape[0]
    )

    for batch_start in range(
        0,
        num_windows,
        batch_size,
    ):

        batch_end = min(
            batch_start + batch_size,
            num_windows,
        )

        raw_states = torch.from_numpy(
            state_windows[
                batch_start:batch_end
            ]
        ).to(device)

        actions = torch.from_numpy(
            action_windows[
                batch_start:batch_end
            ]
        ).to(device)

        states = (
            raw_states
            - state_mean
        ) / state_std

        batch = states.shape[0]

        tssm_state = model.initial(
            batch,
            device=device,
        )

        prev_action = torch.zeros(
            batch,
            model.action_dim,
            device=device,
        )

        deter = None

        # -----------------------------
        # Posterior context warm-up
        # -----------------------------

        for t in range(context):

            (
                tssm_state,
                deter,
                _,
                _,
            ) = model.observe_step(
                tssm_state,
                prev_action,
                states[:, t],
            )

            prev_action = (
                actions[:, t]
            )

        current_state = (
            states[:, context - 1]
        )

        # -----------------------------
        # Prior-only rollout
        # -----------------------------

        for h in range(horizon):

            action_index = (
                context - 1 + h
            )

            action = actions[
                :,
                action_index,
            ]

            (
                tssm_state,
                deter,
                _,
            ) = model.imagine_step(
                tssm_state,
                prev_action,
                deterministic=deterministic,
            )

            predicted_delta = (
                model.predict_delta(
                    deter,
                    tssm_state,
                )
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

            target_index = (
                context + h
            )

            target = (
                raw_states[
                    :,
                    target_index,
                ]
            )

            squared_error = (
                prediction - target
            ).pow(2)

            squared_error_sum[h] += (
                squared_error.sum()
            )

            squared_error_dim_sum[h] += (
                squared_error.sum(dim=0)
            )

        element_count += (
            batch
            * model.state_dim
        )

        sample_count += batch

    mse = (
        squared_error_sum
        / element_count
    )

    rmse = torch.sqrt(mse)

    mse_per_dim = (
        squared_error_dim_sum
        / sample_count
    )

    rmse_per_dim = torch.sqrt(
        mse_per_dim
    )

    return (
        rmse.cpu().numpy(),
        rmse_per_dim.cpu().numpy(),
    )


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"device={device}")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

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

    model = TSSM(
        state_dim=checkpoint[
            "state_dim"
        ],
        action_dim=checkpoint[
            "action_dim"
        ],
        model_dim=checkpoint[
            "model_dim"
        ],
        num_layers=checkpoint[
            "num_layers"
        ],
        num_heads=checkpoint[
            "num_heads"
        ],
        ff_dim=checkpoint[
            "ff_dim"
        ],
        stoch_groups=checkpoint[
            "stoch_groups"
        ],
        classes=checkpoint[
            "classes"
        ],
        hidden_dim=checkpoint[
            "hidden_dim"
        ],
        max_seq_len=checkpoint[
            "max_seq_len"
        ],
        dropout=0.1,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    state_windows, action_windows = load_validation_windows(
        args.episodes,
        context=args.context,
        horizon=args.horizon,
        max_windows=args.max_windows,
        seed=args.seed,
        complete_only=args.complete_only,
    )

    print(
        f"evaluation windows="
        f"{len(state_windows)} "
        f"context={args.context} "
        f"horizon={args.horizon}"
    )

    # ---------------------------------
    # One-step prior diagnostic
    # ---------------------------------

    (
        one_step_rmse,
        one_step_rmse_dims,
    ) = evaluate_one_step_prior(
        model=model,
        state_windows=state_windows,
        action_windows=action_windows,
        context=args.context,
        batch_size=args.batch_size,
        device=device,
        state_mean=state_mean,
        state_std=state_std,
        deterministic=args.deterministic
    )

    print(
        "teacher-forced one-step "
        f"prior RMSE="
        f"{one_step_rmse:.6f}"
    )

    print(
        "per-dimension RMSE="
        f"{one_step_rmse_dims}"
    )

    # ---------------------------------
    # Rollout evaluation
    # ---------------------------------

    rmse, rmse_per_dim = evaluate(
        model=model,
        state_windows=state_windows,
        action_windows=action_windows,
        context=args.context,
        horizon=args.horizon,
        batch_size=args.batch_size,
        device=device,
        state_mean=state_mean,
        state_std=state_std,
        deterministic=args.deterministic
    )

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

    print("\nRollout RMSE:")

    for h in report_horizons:

        if h <= args.horizon:

            print(
                f"h={h:02d} "
                f"rmse="
                f"{rmse[h - 1]:.6f}"
            )

    print(
        "\nPer-dimension rollout RMSE:"
    )

    for h in report_horizons:

        if h <= args.horizon:

            values = (
                rmse_per_dim[h - 1]
            )

            dim_text = " ".join(
                f"dim{i}={value:.6f}"
                for i, value in enumerate(values)
            )

            print(
                f"h={h:02d} "
                f"{dim_text}"
            )

    # ---------------------------------
    # Save CSV
    # ---------------------------------

    args.csv_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        args.csv_out,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        dim_headers = [
            f"dim{i}_rmse"
            for i in range(model.state_dim)
        ]

        writer.writerow(
            [
                "metric",
                "horizon",
                "rmse",
                *dim_headers,
            ]
        )

        writer.writerow(
            [
                "teacher_forced_one_step",
                1,
                float(one_step_rmse),
                *[
                    float(x)
                    for x in one_step_rmse_dims
                ],
            ]
        )

        for h in range(
            1,
            args.horizon + 1,
        ):
            values = rmse_per_dim[h - 1]

            writer.writerow(
                [
                    "rollout",
                    h,
                    float(rmse[h - 1]),
                    *[
                        float(x)
                        for x in values
                    ],
                ]
            )

    print(
        f"\nsaved metrics to {args.csv_out}"
    )


if __name__ == "__main__":
    main()