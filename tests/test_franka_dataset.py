import numpy as np
from torch.utils.data import DataLoader

from src.data.episodes import (
    extract_episodes,
    split_episode_indices,
)

from src.data.dataset import (
    TrueEpisodeSequenceDataset,
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

complete_len = max(
    len(ep["actions"])
    for ep in episodes
)

episodes = [
    ep for ep in episodes
    if len(ep["actions"]) == complete_len
]

train_idx, val_idx = split_episode_indices(
    episodes,
    val_fraction=0.1,
    seed=42,
)

dataset = TrueEpisodeSequenceDataset(
    episodes,
    train_idx,
    seq_len=20,
    stride=5,
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
)

states, actions, commands = next(
    iter(loader)
)

print("states:", states.shape)
print("actions:", actions.shape)
print("commands:", commands.shape)

print()
print("state mean:", states.mean(dim=(0, 1)))
print("command mean:", commands.mean(dim=(0, 1)))