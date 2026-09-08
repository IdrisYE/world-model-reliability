import numpy as np


def extract_episodes(
    states,
    actions,
    dones,
    commands=None,
    min_length=2,
):
    episodes = []

    num_envs = states.shape[0]
    num_steps = actions.shape[1]

    for env_idx in range(num_envs):
        start = 0

        for t in range(num_steps):

            if dones[env_idx, t]:

                # Exclude the done/reset transition itself.
                length = t - start

                if length >= min_length:
                    episode = {
                        "states": states[
                            env_idx,
                            start:t + 1,
                        ].copy(),

                        "actions": actions[
                            env_idx,
                            start:t,
                        ].copy(),
                    }

                    if commands is not None:
                        episode["commands"] = commands[
                            env_idx,
                            start:t + 1,
                        ].copy()

                    episodes.append(episode)

                start = t + 1

        # Final unfinished episode in the stream.
        if start < num_steps:

            length = num_steps - start

            if length >= min_length:
                episode = {
                    "states": states[
                        env_idx,
                        start:num_steps + 1,
                    ].copy(),

                    "actions": actions[
                        env_idx,
                        start:num_steps,
                    ].copy(),
                }

                if commands is not None:
                    episode["commands"] = commands[
                        env_idx,
                        start:num_steps + 1,
                    ].copy()

                episodes.append(episode)

    return episodes


def split_episode_indices(
    episodes,
    val_fraction=0.1,
    seed=42,
):
    rng = np.random.default_rng(seed)

    indices = np.arange(
        len(episodes)
    )

    rng.shuffle(indices)

    num_val = max(
        1,
        int(
            len(indices)
            * val_fraction
        ),
    )

    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    return train_indices, val_indices


def split_episode_indices(
    episodes,
    val_fraction=0.1,
    seed=42,
    ):
        """
        Return reproducible train/validation episode indices.
        """

        rng = np.random.default_rng(seed)

        indices = np.arange(len(episodes))
        rng.shuffle(indices)

        num_val = max(
            1,
            int(len(indices) * val_fraction),
        )

        val_indices = indices[:num_val]
        train_indices = indices[num_val:]

        return train_indices, val_indices
