import argparse
from pathlib import Path

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.episodes import extract_episodes, split_episode_indices

from src.data.dataset import TrueEpisodeSequenceDataset
from src.models.rssm import DreamerV3RSSM, RSSMState


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
        default=Path("checkpoints/rssm.pt"),
    )

    parser.add_argument(
        "--complete-only",
        action="store_true",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--stride", type=int, default=5)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device={device}")

    raw_data = np.load(args.episodes)

    episodes = extract_episodes(
        raw_data["states"],
        raw_data["actions"],
        raw_data["dones"],
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

    state_dim = episodes[0]["states"].shape[-1]
    action_dim = episodes[0]["actions"].shape[-1]

    print(
        f"state_dim={state_dim} "
        f"action_dim={action_dim}"
    )

    train_episode_indices, val_episode_indices = (
        split_episode_indices(
            episodes,
            val_fraction=0.1,
            seed=42,
        )
    )

    print(
        f"total episodes={len(episodes)} \t"
        f"train episodes={len(train_episode_indices)} \t"
        f"val episodes={len(val_episode_indices)}"
    )

    train_set = TrueEpisodeSequenceDataset(
        episodes=episodes,
        episode_indices=train_episode_indices,
        seq_len=args.seq_len,
        stride=args.stride,
    )

    val_set = TrueEpisodeSequenceDataset(
        episodes=episodes,
        episode_indices=val_episode_indices,
        seq_len=args.seq_len,
        stride=args.stride,
        state_mean=train_set.state_mean,
        state_std=train_set.state_std,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(
        f"train windows={len(train_set)} \t"
        f"val windows={len(val_set)}"
    )

    print(f"state mean={train_set.state_mean} ")
    print(f"state std={train_set.state_std} ")

    model = DreamerV3RSSM(
        state_dim=state_dim,
        action_dim=action_dim,
        deter_dim=256,
        hidden_dim=256,
        stoch_groups=32,
        classes=16,
        embed_dim=128,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    best_val = float("inf")

    if state_dim == 4:
        # Preserve the existing Cartpole weighting.
        state_loss_weights = torch.tensor(
            [1.0, 1.5, 1.0, 2.0],
            dtype=torch.float32,
            device=device,
        )
    else:
        # Franka: normalized state dimensions are
        # weighted equally.
        state_loss_weights = torch.ones(
            state_dim,
            dtype=torch.float32,
            device=device,
        )


    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        # ----------------------
        # Train
        # ----------------------

        model.train()

        train_total = 0.0
        train_recon_total = 0.0
        train_prior_total = 0.0
        train_batches = 0

        for states, actions in train_loader:

            states = states.to(device)
            actions = actions.to(device)

            batch_size = states.shape[0]

            optimizer.zero_grad()

            recon_loss = torch.tensor(0.0, device=device)
            prior_pred_loss = torch.tensor(0.0, device=device)
            dyn_loss = torch.tensor(0.0, device=device)
            rep_loss = torch.tensor(0.0, device=device)

            # ---------------------------------
            # Initialize latent state from x_0.
            # No action precedes the first
            # observation in the sampled window.
            # ---------------------------------

            rssm_state = model.initial(
                batch_size,
                device=device,
            )

            zero_action = torch.zeros(
                batch_size,
                model.action_dim,
                device=device,
            )

            rssm_state, _, _ = model.observe_step(
                rssm_state,
                zero_action,
                states[:, 0],
            )

            # ---------------------------------
            # Predict transitions:
            # (x_t, a_t) -> x_{t+1}
            # ---------------------------------

            for t in range(args.seq_len - 1):

                observation = states[:, t]
                next_observation = states[:, t + 1]
                action = actions[:, t]

                # ----------------------
                # Dynamics prior:
                # posterior z_t + a_t
                # -> prior z_{t+1}
                # ----------------------

                prior_state, prior_logits = (
                    model.imagine_step(
                        rssm_state,
                        action,
                    )
                )

                # ----------------------
                # Prior prediction loss
                # ----------------------

                prior_delta = model.predict_state(
                    prior_state
                )

                prior_prediction = (
                    observation + prior_delta
                )

                prior_error = (
                    prior_prediction
                    - next_observation
                ).pow(2)

                prior_pred_loss = (
                    prior_pred_loss
                    + (
                        prior_error
                        * state_loss_weights
                    ).mean()
                )

                # ----------------------
                # Posterior for x_{t+1}
                #
                # Use the SAME deterministic
                # state produced by a_t.
                # Do not advance the GRU twice.
                # ----------------------

                next_embed = model.encoder(
                    next_observation
                )

                posterior_logits = model._posterior(
                    prior_state.deter,
                    next_embed,
                )

                posterior_sample = (
                    model._sample_straight_through(
                        posterior_logits
                    )
                )

                posterior_state = RSSMState(
                    deter=prior_state.deter,
                    stoch=posterior_sample.reshape(
                        batch_size,
                        -1,
                    ),
                )

                # ----------------------
                # Posterior reconstruction
                # ----------------------

                posterior_delta = model.predict_state(
                    posterior_state
                )

                posterior_prediction = (
                    observation + posterior_delta
                )

                recon_error = (
                    posterior_prediction
                    - next_observation
                ).pow(2)

                recon_loss = (
                    recon_loss
                    + (
                        recon_error
                        * state_loss_weights
                    ).mean()
                )

                # ----------------------
                # KL:
                # prior z_{t+1}
                # vs posterior z_{t+1}
                # ----------------------

                dyn_kl, rep_kl = model.kl_losses(
                    prior_logits,
                    posterior_logits,
                )

                dyn_loss = dyn_loss + dyn_kl
                rep_loss = rep_loss + rep_kl

                # Posterior becomes the latent
                # state for the next transition.
                rssm_state = posterior_state


            num_pred_steps = args.seq_len - 1

            recon_loss = recon_loss / num_pred_steps
            prior_pred_loss = prior_pred_loss / num_pred_steps
            dyn_loss = dyn_loss / num_pred_steps
            rep_loss = rep_loss / num_pred_steps

            # DreamerV3-style relative weighting.
            loss = (
                recon_loss
                + prior_pred_loss
                + 0.5 * dyn_loss
                + 0.1 * rep_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=100.0,
            )

            optimizer.step()

            train_total += loss.item()
            train_recon_total += recon_loss.item()
            train_prior_total += prior_pred_loss.item()
            train_batches += 1

        train_loss = train_total / train_batches
        train_recon = train_recon_total / train_batches
        train_prior = train_prior_total / train_batches

        # ----------------------
        # Validation
        # ----------------------

        model.eval()

        val_recon_total = 0.0
        val_prior_total = 0.0
        val_batches = 0

        with torch.no_grad():

            for states, actions in val_loader:

                states = states.to(device)
                actions = actions.to(device)

                batch_size = states.shape[0]

                rssm_state = model.initial(
                    batch_size,
                    device=device,
                )

                zero_action = torch.zeros(
                    batch_size,
                    model.action_dim,
                    device=device,
                )

                # Initialize posterior at x_0.
                rssm_state, _, _ = model.observe_step(
                    rssm_state,
                    zero_action,
                    states[:, 0],
                )

                val_recon_loss = torch.tensor(
                    0.0,
                    device=device,
                )

                val_prior_loss = torch.tensor(
                    0.0,
                    device=device,
                )

                for t in range(args.seq_len - 1):

                    observation = states[:, t]
                    next_observation = states[:, t + 1]
                    action = actions[:, t]

                    # Prior transition using a_t.
                    prior_state, prior_logits = (
                        model.imagine_step(
                            rssm_state,
                            action,
                        )
                    )

                    # ----------------------
                    # Prior prediction
                    # ----------------------

                    prior_delta = model.predict_state(
                        prior_state
                    )

                    prior_prediction = (
                        observation + prior_delta
                    )

                    prior_error = (
                        prior_prediction
                        - next_observation
                    ).pow(2)

                    val_prior_loss = (
                        val_prior_loss
                        + (
                            prior_error
                            * state_loss_weights
                        ).mean()
                    )

                    # ----------------------
                    # Posterior at t+1
                    # ----------------------

                    next_embed = model.encoder(
                        next_observation
                    )

                    posterior_logits = model._posterior(
                        prior_state.deter,
                        next_embed,
                    )

                    posterior_sample = (
                        model._sample_straight_through(
                            posterior_logits
                        )
                    )

                    posterior_state = RSSMState(
                        deter=prior_state.deter,
                        stoch=posterior_sample.reshape(
                            batch_size,
                            -1,
                        ),
                    )

                    # ----------------------
                    # Posterior reconstruction
                    # ----------------------

                    posterior_delta = model.predict_state(
                        posterior_state
                    )

                    posterior_prediction = (
                        observation + posterior_delta
                    )

                    recon_error = (
                        posterior_prediction
                        - next_observation
                    ).pow(2)

                    val_recon_loss = (
                        val_recon_loss
                        + (
                            recon_error
                            * state_loss_weights
                        ).mean()
                    )

                    rssm_state = posterior_state

                num_pred_steps = args.seq_len - 1

                val_recon_loss /= num_pred_steps
                val_prior_loss /= num_pred_steps

                val_recon_total += val_recon_loss.item()
                val_prior_total += val_prior_loss.item()

                val_batches += 1

        val_recon = (
            val_recon_total / val_batches
        )

        val_prior = (
            val_prior_total / val_batches
        )
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start


        print(
            f"epoch={epoch:03d} \t"
            f"time={epoch_time:.2f}s \t"
            f"train_loss={train_loss:.6f} \t"
            f"train_recon={train_recon:.6f} \t"
            f"train_prior={train_prior:.6f}"
        )
        print(
            f"val_recon={val_recon:.6f} \t"
            f"val_prior={val_prior:.6f} \t"
        )

        if val_prior < best_val:
            best_val = val_prior

            args.out.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "state_mean": train_set.state_mean,
                    "state_std": train_set.state_std,
                },
                args.out,
            )

            print(
                f"saved best checkpoint "
                f"val_prior={best_val:.6f}"
            )


if __name__ == "__main__":
    main()