import numpy as np

from src.data.episodes import (
    extract_episodes,
    split_episode_indices,
)

data = np.load("data/cartpole_episodes.npz")

episodes = extract_episodes(
    data["states"],
    data["actions"],
    data["dones"],
)

train_idx, val_idx = split_episode_indices(
    episodes,
    val_fraction=0.1,
    seed=42,
)


def collect_dim2_deltas(indices):
    all_deltas = []
    per_episode_max = []

    for ep_idx in indices:
        states = episodes[ep_idx]["states"]

        deltas = np.diff(states, axis=0)

        dim2 = np.abs(deltas[:, 2])

        all_deltas.append(dim2)
        per_episode_max.append(dim2.max())

    return (
        np.concatenate(all_deltas),
        np.asarray(per_episode_max),
    )


train_delta, train_ep_max = collect_dim2_deltas(train_idx)
val_delta, val_ep_max = collect_dim2_deltas(val_idx)


def report(name, deltas, ep_max):
    print(f"\n{name}")
    print(f"transitions={len(deltas)}")

    print(
        "delta quantiles:",
        np.quantile(
            deltas,
            [0.5, 0.9, 0.95, 0.99, 0.999, 1.0],
        ),
    )

    print(
        "episode max quantiles:",
        np.quantile(
            ep_max,
            [0.5, 0.9, 0.95, 0.99, 1.0],
        ),
    )

    for threshold in [
        0.01,
        0.05,
        0.1,
        0.5,
        1.0,
    ]:
        print(
            f"|delta dim2| > {threshold}: "
            f"{np.sum(deltas > threshold)} "
            f"({100*np.mean(deltas > threshold):.4f}%)"
        )


report(
    "TRAIN",
    train_delta,
    train_ep_max,
)

report(
    "VAL",
    val_delta,
    val_ep_max,
)