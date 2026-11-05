import gymnasium as gym
import numpy as np
from scipy import stats
from stable_baselines3 import SAC 


def run_episode(policy, env, seed):
    obs, _ = env.reset(seed=seed)
    done = False 
    total_reward = 0.0

    while not done: 
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if truncated or terminated:
            done = True
    return total_reward


def confidence_interval(returns):
    n_returns = len(returns)
    mean_return = np.mean(returns)
    standard_deviation = np.std(returns, ddof=1)
    std_error = standard_deviation / np.sqrt(n_returns)
    factor = stats.t.ppf(0.975, n_returns - 1)
    halbbreite = factor * std_error
    return mean_return, halbbreite


def evaluate(policy, env, seeds):
    returns = []
    for s in seeds:
        returns.append(run_episode(policy, env, s))
    mean_return, halbbreite = confidence_interval(returns)
    return mean_return, halbbreite, returns


if __name__ == "__main__":
    env = gym.make("Reacher-v5")
    env.action_space.seed(0)
    policy = lambda obs: env.action_space.sample()
    mean, half, returns = evaluate(policy, env, range(20))
    print(f"ZufallsPolitik: {mean:.2f} ± {half:.2f}")

    model = SAC.load("models/sac_Reacher-v5_seed0")
    expert_policy = lambda obs: model.predict(obs, deterministic=True)[0]
    mean, half, returns = evaluate(expert_policy, env, range(20))
    print(f"Experte: {mean:.2f} ± {half:.2f}")
