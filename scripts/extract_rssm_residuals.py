import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from src.data.episodes import (
    extract_episodes,
    split_episode_indices,
)

from src.models.rssm import (
    DreamerV3RSSM,
    RSSMState,
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
    # Load checkpoint/model
    # -------------------------------------------------

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    state_dim = checkpoint["state_dim"]
    action_dim = checkpoint["action_dim"]

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

    model = DreamerV3RSSM(
        state_dim=state_dim,
        action_dim=action_dim,
        deter_dim=256,
        hidden_dim=256,
        stoch_groups=32,
        classes=16,
        embed_dim=128,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # -------------------------------------------------
    # Output CSV
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

        with torch.no_grad():

            for ep_idx in val_indices:

                episode = episodes[ep_idx]

                states_raw = torch.from_numpy(
                    episode["states"]
                ).float().to(device)

                actions = torch.from_numpy(
                    episode["actions"]
                ).float().to(device)

                states = (
                    states_raw - state_mean
                ) / state_std

                num_steps = len(actions)

                # -------------------------------------
                # Initialize posterior at x_0
                # -------------------------------------

                rssm_state = model.initial(
                    batch_size=1,
                    device=device,
                )

                zero_action = torch.zeros(
                    1,
                    action_dim,
                    device=device,
                )

                rssm_state, _, _ = (
                    model.observe_step(
                        rssm_state,
                        zero_action,
                        states[0].unsqueeze(0),
                    )
                )

                # -------------------------------------
                # Real trajectory filtering
                # -------------------------------------

                for t in range(num_steps):

                    current_state = (
                        states[t].unsqueeze(0)
                    )

                    next_state = (
                        states[t + 1].unsqueeze(0)
                    )

                    next_state_raw = (
                        states_raw[t + 1]
                        .unsqueeze(0)
                    )

                    action = (
                        actions[t].unsqueeze(0)
                    )

                    # ---------------------------------
                    # PRIOR:
                    # posterior z_t + a_t
                    # -> prior z_{t+1}
                    # ---------------------------------

                    prior_state, prior_logits = (
                        model.imagine_step(
                            rssm_state,
                            action,
                        )
                    )

                    prior_delta = (
                        model.predict_state(
                            prior_state
                        )
                    )

                    predicted_next_norm = (
                        current_state
                        + prior_delta
                    )

                    predicted_next_raw = (
                        predicted_next_norm
                        * state_std
                        + state_mean
                    )

                    # ---------------------------------
                    # Residual BEFORE posterior update
                    # ---------------------------------

                    error = (
                        predicted_next_raw
                        - next_state_raw
                    ).squeeze(0)

                    squared_error = error.pow(2)

                    prior_residual = torch.sqrt(
                        squared_error.mean()
                    )

                    position_residual = torch.sqrt(
                        squared_error[:9].mean()
                    )

                    velocity_residual = torch.sqrt(
                        squared_error[9:18].mean()
                    )

                    dim_residuals = error.abs()

                    writer.writerow(
                        [
                            "rssm",
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

                    # ---------------------------------
                    # POSTERIOR CORRECTION:
                    # use real x_{t+1}
                    #
                    # Important:
                    # same deterministic state from
                    # the prior transition.
                    # Do NOT advance GRU again.
                    # ---------------------------------

                    next_embed = model.encoder(
                        next_state
                    )

                    posterior_logits = (
                        model._posterior(
                            prior_state.deter,
                            next_embed,
                        )
                    )

                    posterior_sample = (
                        model._sample_straight_through(
                            posterior_logits
                        )
                    )

                    rssm_state = RSSMState(
                        deter=prior_state.deter,
                        stoch=(
                            posterior_sample.reshape(
                                1,
                                -1,
                            )
                        ),
                    )

    print(
        f"saved {row_count} residuals "
        f"to {args.out}"
    )


if __name__ == "__main__":
    main()