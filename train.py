# """
# train.py — Train the RL agent on AimingEnv (sim mode)

# Usage:
#     python train.py                    # train 500k steps, save model
#     python train.py --steps 1000000    # longer run
#     python train.py --render           # show pygame window while training (slow)
#     python train.py --eval             # evaluate a saved model

# Curriculum schedule (time-based):
#     Phase 1  0%  → CURRICULUM_RAMP_START%  : std_scale = 0.0  (deterministic)
#     Phase 2  CURRICULUM_RAMP_START → 100%  : std_scale ramps linearly 0 → 1
#     At std_scale = 1 the env uses your full measured EMS variance.

#     ❗ REAL VALUES: after running ems_visualise.py on your data, replace the
#        placeholder constants in aiming_env.py (MEAN_PEAK, STD_PEAK, etc.)
#        and adjust CURRICULUM_RAMP_START below if you want a longer or shorter
#        deterministic warm-up phase.
# """

# import argparse
# import os
# import sys
# import numpy as np

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from env.aiming_env import AimingEnv

# try:
#     from stable_baselines3 import PPO
#     from stable_baselines3.common.env_checker import check_env
#     from stable_baselines3.common.callbacks import BaseCallback
#     from stable_baselines3.common.monitor import Monitor
#     import matplotlib
#     matplotlib.use("TkAgg")
#     import matplotlib.pyplot as plt
# except ImportError:
#     print("Install deps:  pip install stable-baselines3 matplotlib")
#     sys.exit(1)


# # ─────────────────────────────────────────────────────────────────────────────
# # Curriculum timing
# # ❗ REAL VALUES: tune these once you have a sense of how quickly the agent
# #    converges in the deterministic phase.
# #
# #    CURRICULUM_RAMP_START : fraction of total training steps before variance
# #                            starts increasing (0.2 = first 20% deterministic)
# #    CURRICULUM_RAMP_END   : fraction of total steps at which std_scale = 1.0
# #                            (0.8 = full variance reached at 80% of training)
# # ─────────────────────────────────────────────────────────────────────────────
# CURRICULUM_RAMP_START = 0.20   # std_scale = 0 until here
# CURRICULUM_RAMP_END   = 0.80   # std_scale = 1 from here onward
# # ─────────────────────────────────────────────────────────────────────────────


# # ── Callbacks ─────────────────────────────────────────────────────────────────

# class PrintCallback(BaseCallback):
#     """Prints training progress every N steps."""

#     def __init__(self, print_every=10_000):
#         super().__init__()
#         self.print_every  = print_every
#         self._ep_rewards  = []
#         self._ep_hits     = []
#         self._ep_stds     = []

#     def _on_step(self):
#         for info in self.locals.get("infos", []):
#             if "episode" in info:
#                 self._ep_rewards.append(info["episode"]["r"])
#             if "hits" in info:
#                 self._ep_hits.append(info["hits"])
#             if "std_scale" in info:
#                 self._ep_stds.append(info["std_scale"])

#         if self.n_calls % self.print_every == 0 and self._ep_rewards:
#             recent_r   = np.mean(self._ep_rewards[-50:])
#             recent_h   = np.mean(self._ep_hits[-50:])   if self._ep_hits  else 0.0
#             recent_std = np.mean(self._ep_stds[-50:])   if self._ep_stds  else 0.0
#             total      = self.model.num_timesteps
#             pct        = 100 * total / self.locals["total_timesteps"]
#             print(
#                 f"  [{pct:5.1f}%] steps={total:,}  "
#                 f"mean_reward={recent_r:.2f}  "
#                 f"mean_hits={recent_h:.1f}  "
#                 f"std_scale={recent_std:.2f}"
#             )
#         return True


# class CurriculumCallback(BaseCallback):
#     """
#     Time-based curriculum: linearly ramps env.std_scale from 0 → 1
#     between CURRICULUM_RAMP_START and CURRICULUM_RAMP_END of total training.

#     How it works:
#         - Before ramp_start : std_scale = 0.0  (agent trains on deterministic
#                               mean movements — learns the policy structure)
#         - During ramp        : std_scale increases linearly each step
#                               (agent progressively encounters more variance)
#         - After ramp_end     : std_scale = 1.0  (full real-world EMS noise)

#     The unwrapped env is accessed via self.training_env.envs[0].env to get
#     through the Monitor wrapper that train() wraps the env in.

#     ❗ If you switch to a VecEnv with multiple parallel envs, iterate over
#        self.training_env.envs and set std_scale on each one.
#     """

#     def __init__(self, total_timesteps, ramp_start=CURRICULUM_RAMP_START,
#                  ramp_end=CURRICULUM_RAMP_END, update_every=500):
#         super().__init__()
#         self.total_timesteps = total_timesteps
#         self.ramp_start      = ramp_start
#         self.ramp_end        = ramp_end
#         self.update_every    = update_every  # update std_scale every N steps

#     def _std_scale_for_step(self, step):
#         """Map a training step to the target std_scale [0, 1]."""
#         frac = step / self.total_timesteps
#         if frac < self.ramp_start:
#             return 0.0
#         if frac >= self.ramp_end:
#             return 1.0
#         # Linear interpolation between ramp_start and ramp_end
#         ramp_frac = (frac - self.ramp_start) / (self.ramp_end - self.ramp_start)
#         return float(ramp_frac)

#     def _on_step(self):
#         if self.n_calls % self.update_every != 0:
#             return True

#         new_scale = self._std_scale_for_step(self.model.num_timesteps)

#         # Unwrap Monitor → AimingEnv and update the knob
#         try:
#             inner_env = self.training_env.envs[0].env   # Monitor → AimingEnv
#             inner_env.std_scale = new_scale
#         except AttributeError:
#             # Fallback if wrapping structure differs
#             pass

#         return True


# class RenderCallback(BaseCallback):
#     """Calls env.render() every N steps to display pygame window during training."""

#     def __init__(self, render_every=10):
#         super().__init__()
#         self.render_every = render_every

#     def _on_step(self):
#         if self.n_calls % self.render_every == 0:
#             self.model.env.render()
#         return True


# # ── Training ──────────────────────────────────────────────────────────────────

# def train(args):
#     print("\n=== AimingEnv — Stochastic EMS Training ===\n")
#     print(f"Curriculum: deterministic for first {CURRICULUM_RAMP_START*100:.0f}% of steps,")
#     print(f"            full variance reached at {CURRICULUM_RAMP_END*100:.0f}% of steps.\n")

#     render_mode = "human" if args.render else None

#     # Start with std_scale=0 — the curriculum callback will ramp it up
#     env = AimingEnv(
#         screen_w=1280,
#         max_steps=300,
#         target_radius=30,
#         render_mode=render_mode,
#         std_scale=0.0,        # curriculum starts deterministic
#         # ❗ REAL VALUES: if you want to override the module-level constants
#         #    for a specific run without editing aiming_env.py, pass them here:
#         #    mean_peak=15.0, std_peak=6.0,
#         #    mean_trough=12.0, std_trough=5.0,
#         #    p_no_response=0.05,
#         #    pulse_duration_ms=800,
#     )
#     env = Monitor(env)

#     print("Checking environment...")
#     check_env(env, warn=True)
#     print("Environment OK.\n")

#     model = PPO(
#         "MlpPolicy",
#         env,
#         verbose=0,
#         learning_rate=3e-4,
#         n_steps=2048,
#         batch_size=64,
#         n_epochs=10,
#         gamma=0.99,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         ent_coef=0.01,
#         tensorboard_log="./tb_logs/",
#     )

#     callbacks = [
#         PrintCallback(print_every=10_000),
#         CurriculumCallback(total_timesteps=args.steps),
#     ]
#     if args.render:
#         callbacks.append(RenderCallback(render_every=10))

#     print(f"Training for {args.steps:,} steps...")
#     if args.render:
#         print("Rendering: pygame window will display (slower training)")
#     print("TensorBoard: tensorboard --logdir ./tb_logs\n")

#     model.learn(
#         total_timesteps=args.steps,
#         callback=callbacks,
#         progress_bar=False,
#     )

#     os.makedirs("models", exist_ok=True)
#     model.save("models/aiming_ppo_stochastic")
#     print("\nModel saved → models/aiming_ppo_stochastic.zip")
#     env.close()


# # ── Evaluation ────────────────────────────────────────────────────────────────

# def evaluate(args):
#     print("\n=== Evaluating saved model ===\n")

#     # Evaluate at full variance (std_scale=1.0) to match real deployment
#     env = AimingEnv(
#         screen_w=1280,
#         max_steps=300,
#         target_radius=30,
#         render_mode="human",
#         std_scale=1.0,
#     )

#     model = PPO.load("models/aiming_ppo_stochastic", env=env)

#     n_eps = 10
#     all_hits    = []
#     all_rewards = []
#     all_dxs     = []

#     for ep in range(n_eps):
#         obs, _    = env.reset()
#         total_r   = 0.0
#         done      = False
#         ep_dxs    = []

#         while not done:
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, info = env.step(int(action))
#             total_r += reward
#             ep_dxs.append(info["actual_dx"])
#             env.render()
#             done = terminated or truncated

#         all_hits.append(info["hits"])
#         all_rewards.append(total_r)
#         all_dxs.extend(ep_dxs)
#         print(
#             f"  ep {ep+1:2d}: reward={total_r:.1f}  "
#             f"hits={info['hits']}  misses={info['misses']}  "
#             f"mean|dx|={np.mean(np.abs(ep_dxs)):.1f}px"
#         )

#     print(f"\nMean reward:    {np.mean(all_rewards):.2f}")
#     print(f"Mean hits/ep:   {np.mean(all_hits):.1f}")
#     print(f"Mean |dx|/step: {np.mean(np.abs(all_dxs)):.1f}px")
#     env.close()


# # ── Entry point ───────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--steps",  type=int, default=100_000,
#                         help="Training timesteps")
#     parser.add_argument("--render", action="store_true",
#                         help="Show pygame window during training (slow)")
#     parser.add_argument("--eval",   action="store_true",
#                         help="Evaluate saved model at full variance")
#     args = parser.parse_args()

#     if args.eval:
#         evaluate(args)
#     else:
#         train(args)


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
from env.aiming_env import AimingEnv

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
        step_px=12,
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