import numpy as np

from src.data.episodes import (
    extract_episodes,
    split_episode_indices,
)

from src.data.dataset import (
    TrueEpisodeSequenceDataset,
)


def test_true_episode_sequence_dataset():
    data = np.load(
        "data/cartpole_episodes.npz"
    )

    episodes = extract_episodes(
        data["states"],
        data["actions"],
        data["dones"],
    )

    train_indices, val_indices = (
        split_episode_indices(
            episodes,
            val_fraction=0.1,
            seed=42,
        )
    )

    train_set = TrueEpisodeSequenceDataset(
        episodes=episodes,
        episode_indices=train_indices,
        seq_len=20,
        stride=5,
    )

    val_set = TrueEpisodeSequenceDataset(
        episodes=episodes,
        episode_indices=val_indices,
        seq_len=20,
        stride=5,
        state_mean=train_set.state_mean,
        state_std=train_set.state_std,
    )

    assert len(train_set) > 0
    assert len(val_set) > 0

    states, actions = train_set[0]

    assert states.shape == (20, 4)
    assert actions.shape == (20, 1)

    assert np.isfinite(
        train_set.state_mean
    ).all()

    assert np.isfinite(
        train_set.state_std
    ).all()

    assert (
        train_set.state_std > 0
    ).all()

    assert np.allclose(
        train_set.state_mean,
        val_set.state_mean,
    )

    assert np.allclose(
        train_set.state_std,
        val_set.state_std,
    )


def test_windows_do_not_cross_episode_boundaries():
    data = np.load(
        "data/cartpole_episodes.npz"
    )

    episodes = extract_episodes(
        data["states"],
        data["actions"],
        data["dones"],
    )

    train_indices, _ = (
        split_episode_indices(
            episodes,
            val_fraction=0.1,
            seed=42,
        )
    )

    dataset = TrueEpisodeSequenceDataset(
        episodes=episodes,
        episode_indices=train_indices,
        seq_len=20,
        stride=5,
    )

    for ep_idx, start in dataset.windows[:1000]:

        episode = episodes[ep_idx]

        assert (
            start + dataset.seq_len
            <= len(episode["actions"])
        )