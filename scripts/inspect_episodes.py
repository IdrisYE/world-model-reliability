import numpy as np

from src.data.episodes import (
    extract_episodes,
    split_episode_indices,
)


data = np.load(
    "data/cartpole_episodes.npz"
)

episodes = extract_episodes(
    data["states"],
    data["actions"],
    data["dones"],
)

train_indices, val_indices = split_episode_indices(
    episodes,
    val_fraction=0.1,
    seed=42,
)

train_lengths = np.array(
    [
        len(episodes[i]["actions"])
        for i in train_indices
    ]
)

val_lengths = np.array(
    [
        len(episodes[i]["actions"])
        for i in val_indices
    ]
)

print(f"total episodes={len(episodes)}")
print(f"train episodes={len(train_indices)}")
print(f"val episodes={len(val_indices)}")

print(
    f"train mean length="
    f"{train_lengths.mean():.2f}"
)

print(
    f"val mean length="
    f"{val_lengths.mean():.2f}"
)