import argparse
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC


def main() -> None:
    parser = argparse.ArgumentParser(description="SAC-Experten trainieren")
    parser.add_argument("--env", default="Pendulum-v1", help="Gymnasium-Umgebung")
    parser.add_argument("--steps", type=int, default=10_000, help="Trainingsschritte")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make(args.env)
    model = SAC("MlpPolicy", env, seed=args.seed, device="cpu", verbose=0)

    print(f"Trainiere SAC auf {args.env} für {args.steps} Schritte (Seed {args.seed}) ...")
    start = time.time()
    model.learn(total_timesteps=args.steps)
    minutes = (time.time() - start) / 60
    print(f"Fertig in {minutes:.1f} Minuten.")

    out = Path("models") / f"sac_{args.env}_seed{args.seed}"
    out.parent.mkdir(exist_ok=True)
    model.save(out)
    print(f"Gespeichert: {out}.zip")
    env.close()


if __name__ == "__main__":
    main()
