import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()

parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Reach-Franka-v0",
)

parser.add_argument(
    "--num-envs",
    type=int,
    default=256,
)

parser.add_argument(
    "--num-steps",
    type=int,
    default=1000,
)

parser.add_argument(
    "--out",
    type=Path,
    default=Path("data/franka_reach_episodes.npz"),
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def extract_observation_parts(obs):
    """
    Policy observation layout:
        0:9    joint positions
        9:18   joint velocities
        18:25  target pose command
        25:32  previous action

    Returns:
        robot_state: [B, 18]
        command:     [B, 7]
    """

    policy_obs = obs["policy"]

    robot_state = policy_obs[:, :18]
    command = policy_obs[:, 18:25]

    return robot_state, command


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
    )

    obs, info = env.reset()

    state, command = extract_observation_parts(obs)

    num_envs = state.shape[0]
    state_dim = state.shape[1]
    command_dim = command.shape[1]
    action_dim = env.action_space.shape[-1]

    print(
        f"num_envs={num_envs} "
        f"state_dim={state_dim} "
        f"command_dim={command_dim} "
        f"action_dim={action_dim}"
    )

    num_envs = state.shape[0]
    state_dim = state.shape[1]
    action_dim = env.action_space.shape[-1]

    print(
        f"num_envs={num_envs} "
        f"state_dim={state_dim} "
        f"action_dim={action_dim}"
    )

    states = np.zeros(
        (
            num_envs,
            args_cli.num_steps + 1,
            state_dim,
        ),
        dtype=np.float32,
    )

    commands = np.zeros(
        (
            num_envs,
            args_cli.num_steps + 1,
            command_dim,
        ),
        dtype=np.float32,
    )

    actions = np.zeros(
        (
            num_envs,
            args_cli.num_steps,
            action_dim,
        ),
        dtype=np.float32,
    )

    dones = np.zeros(
        (
            num_envs,
            args_cli.num_steps,
        ),
        dtype=np.bool_,
    )

    states[:, 0] = (
        state.detach().cpu().numpy()
    )

    commands[:, 0] = (
        command.detach().cpu().numpy()
    )

    for t in range(args_cli.num_steps):

        # Random actions in [-1, 1].
        action = (
            2.0
            * torch.rand(
                num_envs,
                action_dim,
                device=state.device,
            )
            - 1.0
        )

        next_obs, reward, terminated, truncated, info = env.step(
            action
        )

        next_state, next_command = (
            extract_observation_parts(next_obs)
        )

        done = (
            terminated
            | truncated
        )

        actions[:, t] = (
            action.detach().cpu().numpy()
        )

        states[:, t + 1] = (
            next_state.detach().cpu().numpy()
        )

        commands[:, t + 1] = (
            next_command.detach().cpu().numpy()
        )

        dones[:, t] = (
            done.detach().cpu().numpy()
        )

        state = next_state
        command = next_command

        if (t + 1) % 100 == 0:
            print(
                f"step={t + 1}/{args_cli.num_steps} "
                f"done_count={int(done.sum().item())}"
            )

    args_cli.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        args_cli.out,
        states=states,
        commands=commands,
        actions=actions,
        dones=dones,
    )

    print()
    print(
        f"saved dataset to {args_cli.out}"
    )

    print(
        f"states shape={states.shape}"
    )

    print(
        f"commands shape={commands.shape}"
    )

    print(
        f"actions shape={actions.shape}"
    )

    print(
        f"dones shape={dones.shape}"
    )

    print(
        f"total done flags={dones.sum()}"
    )

    env.close()


if __name__ == "__main__":
    main()

    simulation_app.close()