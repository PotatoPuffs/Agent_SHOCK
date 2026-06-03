"""
run.py — Unified entry point for training and deployment.

Replaces train.py and deploy_agent.py. All modes share the same
core loop structure so training and deployment can never drift apart.

Usage:
    # --- TRAINING (gym env, no real hardware) ---
    python run.py --mode train                        # 500k steps, sim EMS
    python run.py --mode train --steps 1000000        # longer run
    python run.py --mode train --render               # show pygame window

    # --- DEPLOY: fully simulated (no browser, no hardware) ---
    python run.py --mode deploy --cnn sim --ems sim

    # --- DEPLOY: real CNN, simulated EMS (test perception without shocking anyone) ---
    python run.py --mode deploy --cnn real --ems sim

    # --- DEPLOY: real CNN, real EMS (live session on a person) ---
    python run.py --mode deploy --cnn real --ems real

    # --- TEST DEPLOY: AimingEnv observer + real/sim EMS (isolate EMS testing) ---
    python run.py --mode test_deploy --ems sim        # test with simulated EMS
    python run.py --mode test_deploy --ems real       # test with real EMS

    # --- EVALUATE saved model in sim ---
    python run.py --mode eval

Component swap matrix:
    --cnn sim   SimulatedCNNObserver  (random static target, no screen capture)
    --cnn real  RealCNNObserver       (teammate's implementation — screen capture)
    --ems sim   SimulatedEMSController (stochastic displacement, prints actions)
    --ems real  RealEMSController      (teammate's implementation — Arduino serial)

All combinations work. The agent model and reward logic are identical
regardless of which components are wired in.
"""

import argparse
import os
import sys
import time
import numpy as np
import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0

from integration.interfacing import (
    SCREEN_W, TARGET_RADIUS, PULSE_DURATION_MS, OBS_SIZE,
)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_PATH = "models/aiming_ppo3"

# Deploy loop frame-rate cap (ceiling). Tune to your game/EMS pacing.
DEPLOY_TARGET_FPS = 30
DEPLOY_FRAME_DT   = 1.0 / DEPLOY_TARGET_FPS

# Curriculum schedule (training only)
CURRICULUM_RAMP_START = 0.20   # std_scale = 0 until this fraction of training
CURRICULUM_RAMP_END   = 0.80   # std_scale = 1 from this fraction onward

ARDUINO_PORT = "/dev/ttyACM0"
ACTION_MAP = {0: "left", 1: "right", 2: "click", 3: "none"}

# ── Component factory ─────────────────────────────────────────────────────────

def make_observer(cnn_mode: str, ems_sim=None):
    """
    Return the appropriate CNN observer.

    Args:
        cnn_mode : 'sim' | 'hsv' | 'real'
        ems_sim  : SimulatedEMSController instance (needed so SimulatedCNNObserver
                   can be passed to it for cursor tracking). Only used when
                   cnn_mode='sim'.
    """
    if cnn_mode == "sim":
        from integration.simulators import SimulatedCNNObserver
        return SimulatedCNNObserver(screen_w=SCREEN_W)

    elif cnn_mode == "hsv":
        # Interim live observer: real screen capture + HSV masking, no CNN.
        # Detected positions are scaled into the SCREEN_W contract space.
        from integration.simulators import HSVBasedObserver
        return HSVBasedObserver(screen_w=SCREEN_W)

    elif cnn_mode == "real":
        # Teammate's implementation — import path agreed in interfacing.py
        # Uncomment when the real observer is ready:
        from integration.cnn_observer import RealCNNObserver
        return RealCNNObserver(screen_w=SCREEN_W)
        raise NotImplementedError(
            "Real CNN observer not yet connected.\n"
            "Implement perception/cnn_observer.py::RealCNNObserver and "
            "uncomment the import in run.py make_observer()."
        )
    else:
        raise ValueError(f"Unknown --cnn mode: {cnn_mode!r}")


def make_ems(ems_mode: str, observer=None):
    """
    Return the appropriate EMS controller.

    Args:
        ems_mode : 'sim' | 'real'
        observer : SimulatedCNNObserver — needed so SimulatedEMSController
                   can update cursor position after each simulated pulse.
                   Pass None when ems_mode='real'.
    """
    if ems_mode == "sim":
        from integration.simulators import SimulatedEMSController
        return SimulatedEMSController(observer=observer)
    
    elif ems_mode == "hsv":
        # Interim live observer: real screen capture + HSV masking, no CNN.
        # Detected positions are scaled into the SCREEN_W contract space.
        from integration.simulators import HSVEMSController
        return HSVEMSController(observer=observer)

    elif ems_mode == "real":
        # Teammate's implementation
        # Uncomment when the real controller is ready:
        from integration.ems_controller import RealEMSController
        return RealEMSController(port=ARDUINO_PORT,  baud=9600)
        raise NotImplementedError(
            "Real EMS controller not yet connected.\n"
            "Implement hardware/ems_controller.py::RealEMSController and "
            "uncomment the import in run.py make_ems()."
        )
    else:
        raise ValueError(f"Unknown --ems mode: {ems_mode!r}")


# ── Shared deploy loop ────────────────────────────────────────────────────────

def deploy_loop(model, observer, ems):
    """
    Core perception → decision → action loop.

    Identical logic regardless of whether CNN/EMS are real or simulated.
    This is the loop that runs against the live aiming.pro game.

    Smart command throttling:
        - Only sends a new command to EMS when the RL action changes
        - Relays maintain state between frames (no redundant serial writes)
        - Loop runs at CNN capture speed (not throttled by EMS)

    Args:
        model    : loaded PPO model
        observer : BaseCNNObserver — get_state() called each frame
        ems      : BaseEMSController — send_action() called only on action change
    """
    # from stable_baselines3 import PPO
    # from integration.interfacing import BaseCNNObserver, BaseEMSController
    from integration.simulators import SimulatedCNNObserver, HSVBasedObserver

    last_dx           = 0.0
    last_action_sent  = None  # Track last EMS command to avoid redundant sends
    frame_count       = 0
    hits              = 0
    is_sim_observer   = isinstance(observer, SimulatedCNNObserver)
    is_hsv_observer   = isinstance(observer, HSVBasedObserver)

    print(f"\n  CNN={'sim' if is_sim_observer else 'hsv' if is_hsv_observer else 'real'}  "
          f"EMS={'sim' if hasattr(ems, 'std_scale') else 'hsv' if is_hsv_observer else'real'}  ")

    try:
        while True:
            t0 = time.perf_counter()

            # 1. Get observation from CNN (real screenshot or sim)
            #    observer is the single source of truth for cursor_x and target_x
            obs, target_x, cursor_x = observer.get_state(
                last_dx=last_dx,
                pulse_duration_ms=PULSE_DURATION_MS,
            )

            # Sanity check obs shape matches what the model was trained on
            assert obs.shape == (OBS_SIZE,), \
                f"Obs shape mismatch: got {obs.shape}, expected ({OBS_SIZE},)"

            # 2. RL agent picks action
            action, _ = model.predict(obs, deterministic=True)
            action_str = ACTION_MAP[int(action)]

            # 3. Send to EMS
            #    - Simulated: always call to get displacement for obs feedback
            #    - Real: only call when action changes (command deduplication)
            if is_sim_observer or hasattr(ems, 'std_scale'):
                # Simulated EMS — returns actual_dx for obs feedback
                actual_dx = ems.send_action(action_str)
                last_dx = actual_dx if actual_dx is not None else 0.0
            else:
                # Real EMS — only send on action change, relay maintains state
                if action_str != last_action_sent:
                    ems.send_action(action_str)
                    print(f"\nSent EMS command: {action_str}")
                    last_action_sent = action_str
                last_dx = 0.0

            # 4. Handle click
            if action_str == "click":
                pixel_error = abs(cursor_x - target_x)
                if pixel_error < TARGET_RADIUS:
                    hits += 1
                    # Tell sim observer to spawn a new target
                    if is_sim_observer:
                        observer.reset_target()

            frame_count += 1
            elapsed = time.perf_counter() - t0

            print(
                f"\r  frame={frame_count:5d}  "
                f"target={target_x:5.0f}px  cursor={cursor_x:5.0f}px  "
                f"err={abs(cursor_x - target_x):4.0f}px  "
                f"act={action_str:<5s}  hits={hits}  "
                f"fps={1/elapsed if elapsed > 0 else 0:.0f}  ",
                end="", flush=True,
            )

            # Frame-rate cap. The real CNN throttles naturally via its screen
            # grab, but the sim/HSV observers return fast enough to free-run and
            # flood the loop — so pace to DEPLOY_TARGET_FPS. This is a ceiling,
            # not a floor: if a frame already took longer, no sleep happens.
            if elapsed < DEPLOY_FRAME_DT:
                time.sleep(DEPLOY_FRAME_DT - elapsed)

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        ems.close()
        if hasattr(observer, "close"):
            observer.close()


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    from env.aiming_env import AimingEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    print("\n=== AimingEnv — Training ===")
    print(f"Curriculum: deterministic for first {CURRICULUM_RAMP_START*100:.0f}% of steps, "
          f"full variance at {CURRICULUM_RAMP_END*100:.0f}%.\n")

    render_mode = "human" if args.render else None

    env = AimingEnv(
        screen_w=SCREEN_W,
        max_steps=300,
        target_radius=TARGET_RADIUS,
        pulse_duration_ms=PULSE_DURATION_MS,
        render_mode=render_mode,
        std_scale=0.0,   # curriculum starts deterministic
    )
    env = Monitor(env)

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

    # ── Callbacks ─────────────────────────────────────────────────────────────

    class PrintCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._ep_rewards = []
            self._ep_hits    = []
            self._ep_stds    = []

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "episode"   in info: self._ep_rewards.append(info["episode"]["r"])
                if "hits"      in info: self._ep_hits.append(info["hits"])
                if "std_scale" in info: self._ep_stds.append(info["std_scale"])

            if self.n_calls % 10_000 == 0 and self._ep_rewards:
                total = self.model.num_timesteps
                pct   = 100 * total / self.locals["total_timesteps"]
                print(
                    f"  [{pct:5.1f}%] steps={total:,}  "
                    f"mean_reward={np.mean(self._ep_rewards[-50:]):.2f}  "
                    f"mean_hits={np.mean(self._ep_hits[-50:]) if self._ep_hits else 0:.1f}  "
                    f"std_scale={np.mean(self._ep_stds[-50:]) if self._ep_stds else 0:.2f}"
                )
            return True

    class CurriculumCallback(BaseCallback):
        """Linearly ramps env.std_scale from 0 -> 1 between ramp_start and ramp_end."""

        def __init__(self, total_timesteps):
            super().__init__()
            self.total_timesteps = total_timesteps

        def _std_scale(self, step):
            frac = step / self.total_timesteps
            if frac < CURRICULUM_RAMP_START:
                return 0.0
            if frac >= CURRICULUM_RAMP_END:
                return 1.0
            return (frac - CURRICULUM_RAMP_START) / (CURRICULUM_RAMP_END - CURRICULUM_RAMP_START)

        def _on_step(self):
            if self.n_calls % 500 == 0:
                scale = self._std_scale(self.model.num_timesteps)
                try:
                    # Unwrap Monitor -> AimingEnv
                    self.training_env.envs[0].env.std_scale = scale
                except AttributeError:
                    pass
            return True

    class RenderCallback(BaseCallback):
        def _on_step(self):
            if self.n_calls % 10 == 0:
                self.model.env.render()
            return True

    callbacks = [PrintCallback(), CurriculumCallback(args.steps)]
    if args.render:
        callbacks.append(RenderCallback())

    print(f"Training for {args.steps:,} steps...")
    print("TensorBoard: tensorboard --logdir ./tb_logs\n")

    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=False,
    )

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}.zip")
    env.close()

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(args):
    """Run saved model in the sim deploy loop for N episodes and print stats."""
    from env.aiming_env import AimingEnv
    from stable_baselines3 import PPO
    from integration.simulators import SimulatedCNNObserver, SimulatedEMSController

    print("\n=== Evaluating saved model (sim, full variance) ===\n")

    observer = SimulatedCNNObserver(screen_w=SCREEN_W)
    ems = SimulatedEMSController(observer=observer, std_scale=1.0)
    
    env = AimingEnv(
        screen_w=SCREEN_W,
        max_steps=300,
        target_radius=TARGET_RADIUS,
        pulse_duration_ms=PULSE_DURATION_MS,
        render_mode="human",
        std_scale=0.0,   # curriculum starts deterministic
    )

    model = PPO.load(MODEL_PATH, env=env)

    n_eps       = 10
    all_hits    = []
    all_rewards = []

    for ep in range(n_eps):
        observer.reset_target()
        last_dx  = 0.0
        total_r  = 0.0
        hits     = 0
        steps    = 0
        max_steps = 300
        

        while steps < max_steps:
            # t0 = time.perf_counter()
            obs, target_x, cursor_x = observer.get_state(
                last_dx=last_dx,
                pulse_duration_ms=PULSE_DURATION_MS,
            )
            # env.render()
            action, _ = model.predict(obs, deterministic=True)
            action_str = ACTION_MAP[int(action)]

            pixel_error = abs(cursor_x - target_x)
            reward = -(pixel_error / SCREEN_W)

            actual_dx = ems.send_action(action_str)
            last_dx   = actual_dx if actual_dx is not None else 0.0

            if action_str == "click":
                if pixel_error < TARGET_RADIUS:
                    reward += 10.0
                    hits   += 1
                    observer.reset_target()
                else:
                    reward -= 2.0

            total_r += reward
            steps += 1
            # elapsed = time.perf_counter() - t0
            # if elapsed < DEPLOY_FRAME_DT:        # DEPLOY_FRAME_DT = 1/30, already defined
            #     time.sleep(DEPLOY_FRAME_DT - elapsed)

        all_hits.append(hits)
        all_rewards.append(total_r)
        print(f"  ep {ep+1:2d}: reward={total_r:.1f}  hits={hits}")

    print(f"\nMean reward:  {np.mean(all_rewards):.2f}")
    print(f"Mean hits/ep: {np.mean(all_hits):.1f}")
    ems.close()

# ── Test Deployment (AimingEnv observer + real/sim EMS) ──────────────────────

def test_deploy(args):
    """
    Test deployment using AimingEnv as the perfect observer (no CNN).
    
    This isolates EMS behavior testing from CNN uncertainty.
    Runs trained RL model against simulated aiming game with real or simulated EMS.
    
    Metrics tracked:
        - Hits per episode
        - Total reward
        - Average error
        - FPS
        - Commands sent to EMS
    """
    from env.aiming_env import AimingEnv
    from stable_baselines3 import PPO
    from integration.simulators import SimulatedEMSController
    from integration.ems_controller import RealEMSController

    print("\n=== Test Deployment (AimingEnv + EMS) ===")
    print(f"EMS: {'sim' if args.ems == 'sim' else 'real'}")
    print(f"Episodes: 10\n")

    # ── Create AimingEnv observer ─────────────────────────────────────────────
    observer = AimingEnv(
        screen_w=SCREEN_W,
        max_steps=300,
        target_radius=TARGET_RADIUS,
        pulse_duration_ms=PULSE_DURATION_MS,
        render_mode="human",
        std_scale=1.0,  # full variance
    )

    # ── Create EMS controller ─────────────────────────────────────────────────
    if args.ems == "sim":
        # Use sim observer's cursor tracking
        from integration.simulators import SimulatedCNNObserver
        sim_observer = SimulatedCNNObserver(screen_w=SCREEN_W)
        ems = SimulatedEMSController(observer=sim_observer, std_scale=1.0)
    else:
        # Real EMS via serial
        ems = RealEMSController(port=ARDUINO_PORT, baud=9600)

    # ── Load model ────────────────────────────────────────────────────────────
    model = PPO.load(MODEL_PATH)

    # ── Metrics collection ────────────────────────────────────────────────────
    n_episodes = 10
    all_episode_hits = []
    all_episode_rewards = []
    total_frames = 0

    print(f"{'Ep':>3} {'Hits':>4} {'Reward':>8} {'Avg Err':>8} {'Frames':>6} {'FPS':>6}")
    print("-" * 50)

    try:
        for ep in range(n_episodes):
            obs, info = observer.reset()
            episode_hits = 0
            episode_reward = 0.0
            episode_errors = []
            last_action_sent = None
            frame_count = 0
            episode_start = time.perf_counter()

            while True:
                t0 = time.perf_counter()

                # ── Get action from RL model ──────────────────────────────────
                action, _ = model.predict(obs, deterministic=True)
                action_str = ACTION_MAP[int(action)]

                # ── Send to EMS (deduplication: only if action changed) ────────
                if action_str != last_action_sent:
                    ems.send_action(action_str)
                    last_action_sent = action_str

                # ── Step environment ─────────────────────────────────────────
                obs, reward, terminated, truncated, info = observer.step(action)
                observer.render()
                done = terminated or truncated

                episode_reward += reward
                frame_count += 1
                total_frames += 1

                # ── Track error ───────────────────────────────────────────────
                # if "error" in info:
                #     episode_errors.append(info["error"])

                if "pixel_error" in info:
                    episode_errors.append(info["pixel_error"])

                # ── Track hits ────────────────────────────────────────────────
                if "hits" in info:
                    episode_hits = info["hits"]

                if done:
                    break

            episode_duration = time.perf_counter() - episode_start
            avg_error = np.mean(episode_errors) if episode_errors else 0.0
            fps = frame_count / max(episode_duration, 0.01)

            all_episode_hits.append(episode_hits)
            all_episode_rewards.append(episode_reward)

            print(f"{ep+1:3d} {episode_hits:4d} {episode_reward:8.2f} "
                  f"{avg_error:8.2f} {frame_count:6d} {fps:6.1f}")

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        observer.close()
        ems.close()

    # ── Summary statistics ────────────────────────────────────────────────────
    print("-" * 50)
    print(f"\nSummary (10 episodes):")
    print(f"  Mean hits/episode:    {np.mean(all_episode_hits):.1f}")
    print(f"  Mean reward/episode:  {np.mean(all_episode_rewards):.2f}")
    print(f"  Total frames:         {total_frames:,}")
    print(f"  EMS type:             {args.ems}")
    print()

# ── Full evaluation ───────────────────────────────────────────────────────────

def _make_heuristic(env):
    """Hand-coded baseline: step toward the target, click when inside radius."""
    sw, r = env.screen_w, env.target_radius
    def policy(obs):
        signed_err_px = obs[0] * sw          # obs[0] = (cursor - target)/screen_w
        if abs(signed_err_px) < r:
            return 2                          # click
        return 0 if signed_err_px > 0 else 1  # cursor right of target -> left, else right
    return policy


def _run_episode(env, policy, max_steps):
    """Run one episode, return per-episode logs. Drives env.step() for reward fidelity."""
    obs, _ = env.reset()
    log = {"reward": 0.0, "click_errors": [], "times_to_hit": [],
           "action_counts": {0: 0, 1: 0, 2: 0}}
    steps_since_spawn = 0
    done = False
    info = {}
    while not done:
        action = int(policy(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        log["reward"] += reward
        log["action_counts"][action] += 1
        steps_since_spawn += 1
        if info.get("hit") or info.get("miss"):
            log["click_errors"].append(info["pixel_error"])   # error at moment of click
        if info.get("hit"):
            log["times_to_hit"].append(steps_since_spawn)      # spawn -> hit latency
            steps_since_spawn = 0                              # new target spawned on hit
        done = terminated or truncated
    log["hits"]       = info.get("hits", 0)
    log["misses"]     = info.get("misses", 0)
    log["total_stim"] = info.get("total_stim", 0)
    return log


def _aggregate_seed(env, policy, n_eps, max_steps):
    """Pool n_eps episodes for one seed into a single metrics dict."""
    rewards, hits = [], []
    hit_tot = miss_tot = stim_tot = 0
    click_errs, tth = [], []
    acts = {0: 0, 1: 0, 2: 0}

    for _ in range(n_eps):
        ep = _run_episode(env, policy, max_steps)
        rewards.append(ep["reward"]); hits.append(ep["hits"])
        hit_tot  += ep["hits"]; miss_tot += ep["misses"]; stim_tot += ep["total_stim"]
        click_errs += ep["click_errors"]; tth += ep["times_to_hit"]
        for k in acts: acts[k] += ep["action_counts"][k]

    clicks = hit_tot + miss_tot
    return {
        "mean_reward":    float(np.mean(rewards)),
        "mean_hits":      float(np.mean(hits)),
        "accuracy":       hit_tot / clicks   if clicks  else 0.0,   # hits / clicks
        "clicks_per_hit": clicks  / hit_tot  if hit_tot else float("nan"),
        "stim_per_hit":   stim_tot / hit_tot if hit_tot else float("nan"),
        "click_err_mean": float(np.mean(click_errs))        if click_errs else float("nan"),
        "time_to_hit":    float(np.mean(tth))               if tth        else float("nan"),
    }


def evaluate_full(args):
    """Multi-seed evaluation with proper metrics + baselines, reported as mean ± std."""
    import random
    from env.aiming_env import AimingEnv
    from stable_baselines3 import PPO

    n_seeds, n_eps, max_steps = args.seeds, args.episodes, 300

    def make_env():
        return AimingEnv(
            screen_w=SCREEN_W, max_steps=max_steps,
            target_radius=TARGET_RADIUS, pulse_duration_ms=PULSE_DURATION_MS,
            std_scale=1.0,   # full variance — evaluate under realistic EMS noise
        )

    model = PPO.load(MODEL_PATH)

    # policy builders: take env, return a callable obs -> action
    policy_builders = {
        "PPO (trained)": lambda env: (lambda obs: int(model.predict(obs, deterministic=True)[0])),
    }
    if args.baselines:
        policy_builders["Random"]    = lambda env: (lambda obs: env.action_space.sample())
        policy_builders["Heuristic"] = _make_heuristic

    print(f"\n=== Full evaluation — {n_seeds} seeds x {n_eps} eps "
          f"({n_seeds * n_eps} episodes/policy, std_scale=1.0) ===\n")

    metric_keys = ["mean_reward", "mean_hits", "accuracy", "clicks_per_hit",
                   "stim_per_hit", "click_err_mean", "time_to_hit"]

    results = {}
    for name, build in policy_builders.items():
        per_seed = []
        for seed in range(n_seeds):
            random.seed(seed); np.random.seed(seed)
            env = make_env()
            env.action_space.seed(seed)
            per_seed.append(_aggregate_seed(env, build(env), n_eps, max_steps))
            env.close()
        # mean ± std across seeds
        results[name] = {k: (np.mean([s[k] for s in per_seed]),
                             np.std([s[k] for s in per_seed])) for k in metric_keys}

    # ── print table ───────────────────────────────────────────────────────────
    labels = {
        "mean_reward": "Reward/ep", "mean_hits": "Hits/ep", "accuracy": "Accuracy",
        "clicks_per_hit": "Clicks/hit", "stim_per_hit": "Pulses/hit",
        "click_err_mean": "ClickErr px", "time_to_hit": "Steps/hit",
    }
    name_w = max(len(n) for n in results)
    header = f"{'Metric':<14}" + "".join(f"{n:>{name_w + 14}}" for n in results)
    print(header); print("-" * len(header))
    for k in metric_keys:
        row = f"{labels[k]:<14}"
        for n in results:
            m, s = results[n][k]
            row += f"{m:>{name_w + 6}.2f} ± {s:<5.2f}"
        print(row)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent SHOCK — unified runner")

    parser.add_argument(
        "--mode", choices=["train", "deploy", "eval", "hsv_test", "test_deploy"], default="train",
        help="train: gym loop | deploy: live game loop | eval: evaluate model | hsv_test: test hsv functionality | test_deploy: test with AimingEnv",
    )
    parser.add_argument(
        "--cnn", choices=["sim", "hsv", "real"], default="sim",
        help="deploy only — sim: fake observer | hsv: live screen capture + HSV masking | real: teammate CNN",
    )
    parser.add_argument(
        "--ems", choices=["sim", "hsv", "real"], default="sim",
        help="deploy only — sim: print actions | hsv: live HSV EMS outputs | real: Arduino serial",
    )
    parser.add_argument(
        "--steps", type=int, default=250_000,
        help="train only — total training timesteps",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="train only — show pygame window (slower)",
    )
    parser.add_argument("--seeds",    type=int, default=5,  help="eval only — number of seeds")
    parser.add_argument("--episodes", type=int, default=10, help="eval only — episodes per seed")
    parser.add_argument("--baselines", action="store_true", help="eval only — include Random + Heuristic baselines")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)

    elif args.mode == "eval":
        if args.baselines:
            evaluate_full(args)
        else:
            evaluate(args)

    elif args.mode == "deploy":
        from stable_baselines3 import PPO
        # Build observer first (sim EMS needs a reference to it for cursor tracking)
        observer = make_observer(args.cnn)
        ems = make_ems(args.ems, observer=observer)
        model = PPO.load(MODEL_PATH)

        print("\n=== Agent SHOCK — Deployment ===")
        deploy_loop(model, observer, ems)

    elif args.mode == "test_deploy":
        test_deploy(args)

    elif args.mode == "hsv_test":
        from integration.simulators import HSVBasedObserver
        observer = HSVBasedObserver(screen_w=SCREEN_W)
        time.sleep(3)
        observer.debug_snapshot()
        
