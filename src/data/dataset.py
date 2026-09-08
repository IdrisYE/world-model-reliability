from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TransitionArrays:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    done: np.ndarray


def save_transitions(path: str | Path, arrays: TransitionArrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        state=arrays.state.astype(np.float32),
        action=arrays.action.astype(np.float32),
        next_state=arrays.next_state.astype(np.float32),
        done=arrays.done.astype(np.bool_),
    )


def load_transitions(path: str | Path) -> TransitionArrays:
    with np.load(Path(path)) as data:
        return TransitionArrays(
            state=data["state"],
            action=data["action"],
            next_state=data["next_state"],
            done=data["done"],
        )


class TransitionDataset(Dataset):
    def __init__(self, path: str | Path):
        arrays = load_transitions(path)
        keep = ~arrays.done.astype(bool)
        self.state = torch.from_numpy(arrays.state[keep]).float()
        self.action = torch.from_numpy(arrays.action[keep]).float()
        self.next_state = torch.from_numpy(arrays.next_state[keep]).float()

    def __len__(self) -> int:
        return self.state.shape[0]

    def __getitem__(self, index: int):
        return self.state[index], self.action[index], self.next_state[index]

class EpisodeSequenceDataset(Dataset):
    def __init__(
        self,
        path,
        seq_len=50,
        stride=1,
        episode_indices=None,
        state_mean=None,
        state_std=None,
    ):
        data = np.load(path)

        self.states = data["states"].astype(np.float32)
        self.actions = data["actions"].astype(np.float32)
        self.dones = data["dones"].astype(np.bool_)

        self.seq_len = seq_len
        self.stride = stride

        if episode_indices is None:
            episode_indices = np.arange(self.states.shape[0])

        self.episode_indices = np.asarray(episode_indices)

        # Normalization statistics must come from training data.
        if state_mean is None or state_std is None:
            train_states = self.states[self.episode_indices]

            state_mean = train_states.reshape(
                -1,
                train_states.shape[-1],
            ).mean(axis=0)

            state_std = train_states.reshape(
                -1,
                train_states.shape[-1],
            ).std(axis=0)

        self.state_mean = np.asarray(
            state_mean,
            dtype=np.float32,
        )

        self.state_std = np.asarray(
            state_std,
            dtype=np.float32,
        )

        self.state_std = np.maximum(
            self.state_std,
            1e-6,
        )

        self.windows = []

        num_steps = self.actions.shape[1]

        for ep in self.episode_indices:
            for start in range(
                0,
                num_steps - seq_len + 1,
                stride,
            ):
                end = start + seq_len

                if self.dones[ep, start:end].any():
                    continue

                self.windows.append(
                    (int(ep), start)
                )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ep, start = self.windows[idx]

        end = start + self.seq_len

        states = self.states[
            ep,
            start:end,
        ].copy()

        actions = self.actions[
            ep,
            start:end,
        ].copy()

        states = (
            states - self.state_mean
        ) / self.state_std

        return (
            torch.from_numpy(states),
            torch.from_numpy(actions),
        )

class TrueEpisodeSequenceDataset(Dataset):
    def __init__(
        self,
        episodes,
        episode_indices,
        seq_len=20,
        stride=5,
        state_mean=None,
        state_std=None,
        command_mean=None,
        command_std=None,
    ):
        self.episodes = episodes
        self.episode_indices = np.asarray(
            episode_indices
        )

        self.seq_len = seq_len
        self.stride = stride

        # ---------------------------------
        # Detect optional command input.
        # ---------------------------------

        self.has_commands = (
            len(self.episode_indices) > 0
            and "commands"
            in episodes[self.episode_indices[0]]
        )

        # ---------------------------------
        # Compute state normalization from
        # selected episodes only.
        # ---------------------------------

        if state_mean is None or state_std is None:

            train_states = np.concatenate(
                [
                    episodes[i]["states"]
                    for i in self.episode_indices
                ],
                axis=0,
            )

            state_mean = train_states.mean(
                axis=0
            )

            state_std = train_states.std(
                axis=0
            )

        self.state_mean = np.asarray(
            state_mean,
            dtype=np.float32,
        )

        self.state_std = np.asarray(
            state_std,
            dtype=np.float32,
        )

        self.state_std = np.maximum(
            self.state_std,
            1e-6,
        )

        # ---------------------------------
        # Compute command normalization
        # from selected episodes only.
        # ---------------------------------

        if self.has_commands:

            if (
                command_mean is None
                or command_std is None
            ):
                train_commands = np.concatenate(
                    [
                        episodes[i]["commands"]
                        for i in self.episode_indices
                    ],
                    axis=0,
                )

                command_mean = train_commands.mean(
                    axis=0
                )

                command_std = train_commands.std(
                    axis=0
                )

            self.command_mean = np.asarray(
                command_mean,
                dtype=np.float32,
            )

            self.command_std = np.asarray(
                command_std,
                dtype=np.float32,
            )

            self.command_std = np.maximum(
                self.command_std,
                1e-6,
            )

        else:
            self.command_mean = None
            self.command_std = None

        # ---------------------------------
        # Build valid windows
        # ---------------------------------

        self.windows = []

        for ep_idx in self.episode_indices:

            episode = episodes[ep_idx]

            actions = episode["actions"]

            num_steps = len(actions)

            if num_steps < seq_len:
                continue

            for start in range(
                0,
                num_steps - seq_len + 1,
                stride,
            ):
                self.windows.append(
                    (
                        int(ep_idx),
                        start,
                    )
                )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):

        ep_idx, start = self.windows[idx]

        episode = self.episodes[ep_idx]

        end = start + self.seq_len

        states = episode["states"][
            start:end
        ].astype(
            np.float32,
            copy=True,
        )

        actions = episode["actions"][
            start:end
        ].astype(
            np.float32,
            copy=True,
        )

        states = (
            states - self.state_mean
        ) / self.state_std

        # Cartpole behavior remains unchanged.
        if not self.has_commands:
            return (
                torch.from_numpy(states),
                torch.from_numpy(actions),
            )

        commands = episode["commands"][
            start:end
        ].astype(
            np.float32,
            copy=True,
        )

        commands = (
            commands - self.command_mean
        ) / self.command_std

        return (
            torch.from_numpy(states),
            torch.from_numpy(actions),
            torch.from_numpy(commands),
        )