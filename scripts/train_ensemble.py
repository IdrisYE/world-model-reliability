import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

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
        "--out",
        type=Path,
        default=Path("checkpoints/ensemble.pt"),
    )

    parser.add_argument(
        "--complete-only",
        action="store_true",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)

    return parser.parse_args()


def gaussian_nll(
    mean,
    logvar,
    target,
):
    inv_var = torch.exp(-logvar)

    loss = (
        (target - mean).pow(2) * inv_var
        + logvar
    )

    return loss.mean()

def make_transitions(
    episodes,
    episode_indices,
    min_length=20,
):
    state_list = []
    action_list = []
    next_state_list = []

    for ep_idx in episode_indices:
        episode = episodes[ep_idx]
        states = episode["states"]
        actions = episode["actions"]

        if len(actions) < min_length:
            continue

        for t in range(len(actions)):
            state_list.append(states[t])
            action_list.append(actions[t])
            next_state_list.append(states[t + 1])

    return (
        np.asarray(state_list, dtype=np.float32),
        np.asarray(action_list, dtype=np.float32),
        np.asarray(next_state_list, dtype=np.float32),
    )

def main():
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"device={device}")
    rng = np.random.default_rng(42)

    # -------------------------------------------------
    # Load episode-major data
    # -------------------------------------------------

    data = np.load(args.episodes)

    states_all = data["states"].astype(np.float32)
    actions_all = data["actions"].astype(np.float32)
    dones_all = data["dones"].astype(np.bool_)

    print(f"states shape={states_all.shape}")
    print(f"actions shape={actions_all.shape}")
    print(f"dones shape={dones_all.shape}")

    episodes = extract_episodes(
        states_all,
        actions_all,
        dones_all,
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

    train_episode_indices, val_episode_indices = (
        split_episode_indices(
            episodes,
            val_fraction=0.1,
            seed=42,
        )
    )

    print(
        f"total episodes={len(episodes)} "
        f"train episodes={len(train_episode_indices)} "
        f"val episodes={len(val_episode_indices)}"
    )

    (
        train_states_raw,
        train_actions,
        train_next_states_raw,
    ) = make_transitions(
        episodes,
        train_episode_indices,
    )

    (
        val_states_raw,
        val_actions,
        val_next_states_raw,
    ) = make_transitions(
        episodes,
        val_episode_indices,
    )

    print(
        f"train transitions={len(train_states_raw)} "
        f"val transitions={len(val_states_raw)}"
    )

    # -------------------------------------------------
    # State normalization using TRAIN only
    # -------------------------------------------------

    train_episode_states = np.concatenate(
        [
            episodes[i]["states"]
            for i in train_episode_indices
        ],
        axis=0,
    )

    state_mean = train_episode_states.mean(axis=0)
                                           
    state_std = train_episode_states.std(axis=0)
    state_std = np.maximum(state_std, 1e-6)

    print(
        f"state mean={state_mean}"
    )

    print(
        f"state std={state_std}"
    )

    train_states = (
        train_states_raw
        - state_mean
    ) / state_std

    train_next_states = (
        train_next_states_raw
        - state_mean
    ) / state_std

    val_states = (
        val_states_raw
        - state_mean
    ) / state_std

    val_next_states = (
        val_next_states_raw
        - state_mean
    ) / state_std

    # -------------------------------------------------
    # Delta targets
    # -------------------------------------------------

    train_deltas = (
        train_next_states
        - train_states
    )

    val_deltas = (
        val_next_states
        - val_states
    )

    print("train delta mean:", train_deltas.mean(axis=0))
    print("train delta std: ", train_deltas.std(axis=0))

    print("val delta mean:  ", val_deltas.mean(axis=0))
    print("val delta std:   ", val_deltas.std(axis=0))

    zero_baseline_rmse = np.sqrt(
        np.mean(val_deltas ** 2)
    )

    zero_baseline_per_dim = np.sqrt(
        np.mean(val_deltas ** 2, axis=0)
    )

    print(
        f"zero-delta baseline RMSE="
        f"{zero_baseline_rmse:.6f}"
    )

    print(
        "zero-delta baseline per-dim RMSE=",
        zero_baseline_per_dim,
    )

    # -------------------------------------------------
    # Validation tensors
    # -------------------------------------------------

    val_states_t = torch.from_numpy(
        val_states
    ).to(device)

    val_actions_t = torch.from_numpy(
        val_actions
    ).to(device)

    val_deltas_t = torch.from_numpy(
        val_deltas
    ).to(device)

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = ProbabilisticEnsemble(
        state_dim=states_all.shape[-1],
        action_dim=actions_all.shape[-1],
        ensemble_size=args.ensemble_size,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # Each member has its own optimizer.
    optimizers = [
        torch.optim.AdamW(
            member.parameters(),
            lr=args.lr,
        )
        for member in model.members
    ]

    best_val = float("inf")

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        epoch_start = (
            time.perf_counter()
        )

        model.train()

        member_train_losses = []

        # ---------------------------------------------
        # Train each ensemble member on an independent
        # bootstrap resample.
        # ---------------------------------------------

        for member_idx, member in enumerate(
            model.members
        ):

            bootstrap_indices = (
                rng.integers(
                    low=0,
                    high=len(
                        train_states
                    ),
                    size=len(
                        train_states
                    ),
                )
            )

            boot_states = (
                torch.from_numpy(
                    train_states[
                        bootstrap_indices
                    ]
                )
            )

            boot_actions = (
                torch.from_numpy(
                    train_actions[
                        bootstrap_indices
                    ]
                )
            )

            boot_deltas = (
                torch.from_numpy(
                    train_deltas[
                        bootstrap_indices
                    ]
                )
            )

            dataset = TensorDataset(
                boot_states,
                boot_actions,
                boot_deltas,
            )

            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
            )

            optimizer = (
                optimizers[
                    member_idx
                ]
            )

            running_loss = 0.0
            batches = 0

            for (
                state,
                action,
                target_delta,
            ) in loader:

                state = state.to(
                    device
                )

                action = action.to(
                    device
                )

                target_delta = (
                    target_delta.to(
                        device
                    )
                )

                optimizer.zero_grad()

                mean, logvar = member(
                    state,
                    action,
                )

                loss = gaussian_nll(
                    mean,
                    logvar,
                    target_delta,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    member.parameters(),
                    max_norm=100.0,
                )

                optimizer.step()

                running_loss += (
                    loss.item()
                )

                batches += 1

            member_train_losses.append(
                running_loss
                / batches
            )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        model.eval()

        val_nlls = []
        val_rmses = []

        with torch.no_grad():

            for member in model.members:

                mean, logvar = member(
                    val_states_t,
                    val_actions_t,
                )

                val_nll = gaussian_nll(
                    mean,
                    logvar,
                    val_deltas_t,
                )

                val_rmse = torch.sqrt(
                    torch.mean(
                        (
                            mean
                            - val_deltas_t
                        ).pow(2)
                    )
                )

                val_nlls.append(
                    val_nll.item()
                )

                val_rmses.append(
                    val_rmse.item()
                )

        mean_train_nll = float(
            np.mean(
                member_train_losses
            )
        )

        mean_val_nll = float(
            np.mean(
                val_nlls
            )
        )

        mean_val_rmse = float(
            np.mean(
                val_rmses
            )
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        print(
            f"epoch={epoch:03d} "
            f"time={epoch_time:.2f}s "
            f"train_nll="
            f"{mean_train_nll:.6f} "
            f"val_nll="
            f"{mean_val_nll:.6f} "
            f"val_delta_rmse="
            f"{mean_val_rmse:.6f}"
        )

        # -------------------------------------------------
        # Checkpoint on predictive RMSE
        # -------------------------------------------------

        if mean_val_rmse < best_val:

            best_val = (
                mean_val_rmse
            )

            args.out.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "state_dim":
                        states_all.shape[-1],

                    "action_dim":
                        actions_all.shape[-1],

                    "ensemble_size":
                        args.ensemble_size,

                    "hidden_dim":
                        args.hidden_dim,

                    "state_mean":
                        state_mean,

                    "state_std":
                        state_std,

                    "train_episode_indices":
                        train_episode_indices,

                    "val_episode_indices":
                        val_episode_indices,
                },
                args.out,
            )

            print(
                "saved best checkpoint "
                f"val_delta_rmse="
                f"{best_val:.6f}"
            )


if __name__ == "__main__":
    main()