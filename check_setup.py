import gymnasium as gym
import numpy as np

ENV_IDS = ["Pendulum-v1", "Reacher-v5"]


def run_random_episode(env_id: str, seed: int = 0) -> dict:
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)

    total_reward, steps, done = 0.0, 0, False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    info = {
        "steps": steps,
        "return": total_reward,
        "obs_shape": env.observation_space.shape,
        "action_shape": env.action_space.shape,
    }
    env.close()
    return info


def main() -> None:
    print(f"gymnasium {gym.__version__}, numpy {np.__version__}\n")
    for env_id in ENV_IDS:
        try:
            r = run_random_episode(env_id)
            print(
                f"OK    {env_id:<12} Schritte: {r['steps']:>3} | "
                f"Return (Zufall): {r['return']:8.1f} | "
                f"Beobachtung: {r['obs_shape']} | Aktion: {r['action_shape']}"
            )
        except Exception as e:  # bewusst breit: wir wollen jeden Fehler sehen
            print(f"FEHLER {env_id}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
