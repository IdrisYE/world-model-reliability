import argparse
import gymnasium as gym

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Reach-Franka-v0",
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
    )

    print("observation_space:")
    print(env.observation_space)

    print("\naction_space:")
    print(env.action_space)

    obs, info = env.reset()

    print("\nobservation type:")
    print(type(obs))

    print("\nobservation:")
    print(obs)

    print("\naction shape:")
    print(env.action_space.shape)

    env.close()


if __name__ == "__main__":
    main()

    simulation_app.close()