"""
train.py — Train the RL agent on AimingEnv (sim mode)

Usage:
    python train.py                    # train 500k steps, save model
    python train.py --steps 1000000   # longer run
    python train.py --render           # show pygame window while training (slow)
    python train.py --eval             # evaluate a saved model
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.aiming_env import AimingEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")  # headless; switch to TkAgg if you want a live window
except ImportError:
    print("Install deps:  pip install stable-baselines3 matplotlib")
    sys.exit(1)


# ─── Callback: live reward curve ──────────────────────────────────────────────

class LivePlotCallback(BaseCallback):
    """Saves a reward plot every N steps to ./training_curve.png"""

    def __init__(self, plot_every=5000, save_path="./training_curve.png"):
        super().__init__()
        self.plot_every = plot_every
        self.save_path = save_path
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_hits = []
        self._ep_reward_buf = []

    def _on_step(self):
        # Collect episode rewards from Monitor wrapper
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
            if "hits" in info:
                self.ep_hits.append(info["hits"])

        if self.n_calls % self.plot_every == 0 and len(self.ep_rewards) > 10:
            self._save_plot()

        return True

    def _save_plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#0D1117")

        def smooth(arr, w=20):
            if len(arr) < w:
                return arr
            return np.convolve(arr, np.ones(w) / w, mode="valid")

        # Reward plot
        ax = axes[0]
        ax.set_facecolor("#161B22")
        r = self.ep_rewards
        ax.plot(r, alpha=0.25, color="#3B82F6", linewidth=0.5)
        ax.plot(smooth(r), color="#60A5FA", linewidth=1.5, label="smoothed reward")
        ax.axhline(0, color="#374151", linewidth=0.5)
        ax.set_title("Episode reward", color="white", fontsize=11)
        ax.set_xlabel("Episode", color="#9CA3AF")
        ax.tick_params(colors="#9CA3AF")
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151")
        ax.legend(labelcolor="white", facecolor="#1F2937")

        # Hits per episode
        ax2 = axes[1]
        ax2.set_facecolor("#161B22")
        if self.ep_hits:
            ax2.plot(self.ep_hits, alpha=0.3, color="#10B981", linewidth=0.5)
            ax2.plot(smooth(self.ep_hits), color="#34D399", linewidth=1.5, label="hits/ep")
        ax2.set_title("Hits per episode", color="white", fontsize=11)
        ax2.set_xlabel("Episode", color="#9CA3AF")
        ax2.tick_params(colors="#9CA3AF")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#374151")
        ax2.legend(labelcolor="white", facecolor="#1F2937")

        plt.tight_layout()
        plt.savefig(self.save_path, dpi=120, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  [plot] saved → {self.save_path}  (ep {len(self.ep_rewards)})")


# ─── Callback: print progress ─────────────────────────────────────────────────

class PrintCallback(BaseCallback):
    def __init__(self, print_every=10000):
        super().__init__()
        self.print_every = print_every
        self._ep_rewards = []
        self._ep_hits = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])
            if "hits" in info:
                self._ep_hits.append(info["hits"])

        if self.n_calls % self.print_every == 0 and self._ep_rewards:
            recent_r = np.mean(self._ep_rewards[-50:])
            recent_h = np.mean(self._ep_hits[-50:]) if self._ep_hits else 0
            total = self.model.num_timesteps
            pct = 100 * total / self.locals["total_timesteps"]
            print(f"  [{pct:5.1f}%] steps={total:,}  "
                  f"mean_reward={recent_r:.2f}  "
                  f"mean_hits={recent_h:.1f}")
        return True


# ─── Training ─────────────────────────────────────────────────────────────────

def train(args):
    print("\n=== AimingEnv — Sim Training ===\n")

    render_mode = "human" if args.render else None
    env = AimingEnv(
        mode="sim",
        screen_w=1280,
        step_px=12,          # pixels per action — tune this
        max_steps=300,
        target_radius=30,
        render_mode=render_mode,
    )
    env = Monitor(env)  # wraps env to log episode stats

    # Sanity-check the env API
    print("Checking environment...")
    check_env(env, warn=True)
    print("Environment OK.\n")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,       # entropy bonus: keeps agent exploring click timing
        tensorboard_log="./tb_logs/",
    )

    callbacks = [
        LivePlotCallback(plot_every=5000, save_path="./training_curve.png"),
        PrintCallback(print_every=10000),
    ]

    print(f"Training for {args.steps:,} steps...")
    print("  → Reward curve: ./training_curve.png  (updated every 5k steps)")
    print("  → TensorBoard:  tensorboard --logdir ./tb_logs\n")

    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=False,
    )

    os.makedirs("./models", exist_ok=True)
    model.save("./models/aiming_ppo")
    print("\nModel saved → ./models/aiming_ppo.zip")
    env.close()


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(args):
    print("\n=== Evaluating saved model ===\n")

    env = AimingEnv(
        mode="sim",
        screen_w=1280,
        step_px=12,
        max_steps=300,
        target_radius=30,
        render_mode="human",
    )

    model = PPO.load("./models/aiming_ppo", env=env)

    n_eps = 10
    all_hits = []
    all_rewards = []

    for ep in range(n_eps):
        obs, _ = env.reset()
        total_r = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            env.render()
            done = terminated or truncated
        all_hits.append(info["hits"])
        all_rewards.append(total_r)
        print(f"  ep {ep+1:2d}: reward={total_r:.1f}  hits={info['hits']}  misses={info['misses']}")

    print(f"\nMean reward: {np.mean(all_rewards):.2f}")
    print(f"Mean hits/ep: {np.mean(all_hits):.1f}")
    env.close()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000, help="Training timesteps")
    parser.add_argument("--render", action="store_true", help="Show pygame window")
    parser.add_argument("--eval", action="store_true", help="Evaluate saved model")
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)