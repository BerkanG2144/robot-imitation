import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC

from evaluate import evaluate

def collect_demos(expert, env, num_episodes, start_seed=1000):
    obs_list = []
    action_list = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=start_seed + ep)
        done = False 
        while not done:
            action, _ = expert.predict(obs, deterministic=True)
            obs_list.append(obs)
            action_list.append(action)

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    return np.array(obs_list, dtype=np.float32), np.array(action_list, dtype=np.float32)

class BCPolicyNet(nn.Module):
    def __init__(self, obs_dim=10, act_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

def train_bc(obs_np, acts_np, epochs=300, lr=1e-3, log_every=100):
    model = BCPolicyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    obs_tensor = torch.from_numpy(obs_np)
    acts_tensor = torch.from_numpy(acts_np)

    loss_history = []
    for epoch in range(1, epochs + 1):
        pred_acts = model(obs_tensor)
        loss = criterion(pred_acts, acts_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if log_every and epoch % log_every == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | MSE Loss: {loss_val:.6f}")

    return model, loss_history


def make_policy_wrapper(bc_model):
    bc_model.eval()

    def policy(obs):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            action_t = bc_model(obs_t)
            return action_t.squeeze(0).numpy()

    return policy


if __name__ == "__main__":
    env = gym.make("Reacher-v5")
    expert = SAC.load("models/sac_Reacher-v5_seed0")

    demo_counts = [1, 5, 10, 20, 50]
    results = []

    print("Baselines -> Zufall: -43.90 | Experte: -3.40\n")

    for n_demos in demo_counts:
        print(f"=== Training mit {n_demos} Demo-Episoden ===")
        # Für jedes Experiment eigenen Seed setzen, damit Initialisierung vergleichbar bleibt
        torch.manual_seed(0)

        obs_data, acts_data = collect_demos(
            expert, env, num_episodes=n_demos, start_seed=1000
        )
        bc_net, losses = train_bc(obs_data, acts_data, epochs=300, lr=1e-3, log_every=100)

        bc_policy = make_policy_wrapper(bc_net)
        mean, half, _ = evaluate(bc_policy, env, range(20))

        results.append((n_demos, len(obs_data), losses[-1], mean, half))
        print(f"-> Ergebnis ({n_demos} Demos): {mean:.2f} ± {half:.2f}\n")

    # Kompakte Übersicht am Ende
    print("\n--- Zusammenfassung ---")
    print(f"{'Demos':>5} | {'Samples':>7} | {'Final MSE':>10} | {'Return (Mean ± 95% KI)':>24}")
    print("-" * 56)
    print(f"{'Rand':>5} | {'-':>7} | {'-':>10} | {'-43.90':>24}")
    for n_demos, n_samples, final_loss, mean, half in results:
        print(f"{n_demos:5d} | {n_samples:7d} | {final_loss:10.6f} | {mean:10.2f} ± {half:6.2f}")
    print(f"{'Exp':>5} | {'-':>7} | {'-':>10} | {'-3.40':>24}")