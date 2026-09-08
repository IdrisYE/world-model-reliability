from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Isaac Lab 2026 launcher/config helpers.
import gymnasium as gym
import isaaclab_tasks  # noqa: F401  # registers task IDs
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect direct-state Cartpole trajectories from Isaac Lab.")
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-Direct-v0")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000, help="Collection steps per parallel environment.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-std", type=float, default=0.35)
    parser.add_argument("--out", type=Path, default=Path("data/cartpole_episodes.npz"))
    add_launcher_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + hydra_args

    env_cfg, _ = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        obs_dict, _ = env.reset(seed=args.seed)
        obs = obs_dict["policy"]

        # The direct Cartpole policy observation is 4D proprioception:
        # cart position, pole angle, cart velocity, pole angular velocity.
        states = [obs.detach().cpu().numpy().copy()]
        actions = []
        dones = []

        for _ in range(args.steps):
            # Broad exploratory behavior for a dynamics dataset. Actions are clipped
            # to the environment action range; env.action_scale maps them to force.
            action = torch.randn((env.num_envs, 1), device=env.device) * args.action_std
            action = action.clamp(-1.0, 1.0)
            next_obs_dict, _, terminated, truncated, _ = env.step(action)
            next_obs = next_obs_dict["policy"]
            done = terminated | truncated

            actions.append(action.detach().cpu().numpy().copy())
            dones.append(done.detach().cpu().numpy().copy())
            states.append(next_obs.detach().cpu().numpy().copy())
            obs = next_obs

        print(f"Collected {args.steps} steps. Saving dataset...")

        # Isaac arrays arrive [T,N,...]; save episode-major [N,T,...].
        states_np = np.stack(states, axis=0).transpose(1, 0, 2).astype(np.float32)
        actions_np = np.stack(actions, axis=0).transpose(1, 0, 2).astype(np.float32)
        dones_np = np.stack(dones, axis=0).transpose(1, 0).astype(np.bool_)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out,
            states=states_np,
            actions=actions_np,
            dones=dones_np,
        )

        flat_out = args.out.with_name(
            args.out.stem + "_transitions.npz"
        )

        s = states_np[:, :-1].reshape(-1, states_np.shape[-1])
        ns = states_np[:, 1:].reshape(-1, states_np.shape[-1])
        a = actions_np.reshape(-1, actions_np.shape[-1])
        d = dones_np.reshape(-1)

        np.savez_compressed(
            flat_out,
            state=s,
            action=a,
            next_state=ns,
            done=d,
        )

        print(f"Saved episodes: {args.out}")
        print(f"Saved transitions: {flat_out}")

        env.close()


if __name__ == "__main__":
    main()
