import numpy as np

from src.data.episodes import (
    extract_episodes,
    split_episode_indices,
)


data = np.load(
    "data/franka_reach_episodes.npz"
)

episodes = extract_episodes(
    data["states"],
    data["actions"],
    data["dones"],
    commands=data["commands"],
)

lengths = np.array(
    [
        len(ep["actions"])
        for ep in episodes
    ]
)

print(
    f"true episodes={len(episodes)}"
)

print(
    f"min length={lengths.min()}"
)

print(
    f"max length={lengths.max()}"
)

print(
    f"mean length={lengths.mean():.2f}"
)

print(
    f"median length={np.median(lengths):.2f}"
)

print(
    "unique lengths/counts=",
    np.unique(
        lengths,
        return_counts=True,
    ),
)

complete_length = lengths.max()

complete_episodes = [
    ep
    for ep in episodes
    if len(ep["actions"])
    == complete_length
]

print()
print(
    f"complete episodes="
    f"{len(complete_episodes)}"
)

train_idx, val_idx = split_episode_indices(
    complete_episodes,
    val_fraction=0.1,
    seed=42,
)

print(
    f"train episodes={len(train_idx)} "
    f"val episodes={len(val_idx)}"
)

# Sanity-check command lengths.
for i, ep in enumerate(
    complete_episodes[:3]
):
    print(
        f"episode {i}: "
        f"states={ep['states'].shape} "
        f"commands={ep['commands'].shape} "
        f"actions={ep['actions'].shape}"
    )