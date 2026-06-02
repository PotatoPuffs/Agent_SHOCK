"""
train.py — Train the RL agent on AimingEnv (sim mode)

Usage:
    python train.py                    # train 500k steps, save model
    python train.py --steps 1000000    # longer run
    python train.py --render           # show pygame window while training (slow)
    python train.py --eval             # evaluate a saved model
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.aiming_env_OLD import AimingEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("TkAgg")  # interactive backend for live windows
except ImportError:
    print("Install deps:  pip install stable-baselines3 matplotlib")
    sys.exit(1)

# Callback: print progress

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


class RenderCallback(BaseCallback):
    """Calls env.render() every N steps to display pygame window during training."""
    def __init__(self, render_every=10):
        super().__init__()
        self.render_every = render_every

    def _on_step(self):
        if self.n_calls % self.render_every == 0:
            self.model.env.render()
        return True

# Training

def train(args):
    print("\n=== AimingEnv — Sim Training ===\n")

    render_mode = "human" if args.render else None
    env = AimingEnv(
        screen_w=1280,
        step_px=12, # pixels per action — tune this
        max_steps=300,
        target_radius=30,
        render_mode=render_mode,
    )
    env = Monitor(env) # wraps env to log episode stats

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
        ent_coef=0.01,
        tensorboard_log="./tb_logs/",
    )

    callbacks = [
        PrintCallback(print_every=10000),
    ]
    if args.render:
        callbacks.append(RenderCallback(render_every=10))

    print(f"Training for {args.steps:,} steps...")
    if args.render:
        print("Rendering: pygame window will display (slower training)")
    print("TensorBoard: tensorboard --logdir ./tb_logs\n")

    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=False,
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/aiming_ppo_render")
    print("\nModel saved → models/aiming_ppo_render.zip")
    env.close()


# Evaluation

def evaluate(args):
    print("\n=== Evaluating saved model ===\n")

    env = AimingEnv(
        screen_w=1280,
        # step_px=12,
        max_steps=300,
        target_radius=30,
        render_mode="human",
    )

    model = PPO.load("models/aiming_ppo_render", env=env)

    n_eps = 10
    all_hits = []
    all_rewards = []

    for ep in range(n_eps):
        obs, _ = env.reset()
        total_r = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_r += reward
            env.render()
            done = terminated or truncated
        all_hits.append(info["hits"])
        all_rewards.append(total_r)
        print(f"  ep {ep+1:2d}: reward={total_r:.1f}  hits={info['hits']}  misses={info['misses']}")

    print(f"\nMean reward: {np.mean(all_rewards):.2f}")
    print(f"Mean hits/ep: {np.mean(all_hits):.1f}")
    env.close()


# Entry point 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250_000, help="Training timesteps")
    parser.add_argument("--render", action="store_true", help="Show pygame window")
    parser.add_argument("--eval", action="store_true", help="Evaluate saved model")
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)