import numpy as np

data = np.load("data/cartpole_episodes.npz")

states = data["states"]
actions = data["actions"]
dones = data["dones"]

threshold = 0.1

for env_idx in range(states.shape[0]):
    for t in range(actions.shape[1]):

        delta = states[env_idx, t + 1, 2] - states[env_idx, t, 2]

        if abs(delta) > threshold:

            print("\n" + "=" * 70)
            print(
                f"env={env_idx} t={t} "
                f"delta_dim2={delta:.6f}"
            )

            print(
                f"done[t]={dones[env_idx, t]}"
            )

            if t > 0:
                print(
                    f"done[t-1]={dones[env_idx, t - 1]}"
                )

            if t + 1 < dones.shape[1]:
                print(
                    f"done[t+1]={dones[env_idx, t + 1]}"
                )

            start = max(0, t - 2)
            end = min(
                states.shape[1],
                t + 4,
            )

            print("\nNearby states:")

            for k in range(start, end):
                print(
                    f"k={k:04d} "
                    f"state={states[env_idx, k]}"
                )