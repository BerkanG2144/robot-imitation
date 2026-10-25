"""Roboterarm in einem Fenster ansehen.

Ohne --model führt der Arm zufällige Aktionen aus. Mit --model spielt ein
trainierter SAC-Experte (siehe train_expert.py).

Aufrufbeispiele (im Projektordner, mit aktiviertem venv):
    python watch.py
    python watch.py --model models/sac_Reacher-v5_seed0
"""

import argparse

import gymnasium as gym


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulation ansehen")
    parser.add_argument("--env", default="Reacher-v5")
    parser.add_argument("--model", default=None, help="Pfad zu einem gespeicherten SAC-Modell (ohne .zip)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make(args.env, render_mode="human")

    model = None
    if args.model:
        from stable_baselines3 import SAC

        model = SAC.load(args.model, device="cpu")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done, total = False, 0.0
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        print(f"Episode {ep + 1}: Return {total:.1f}")

    env.close()


if __name__ == "__main__":
    main()
