# """
# aiming_env.py — Simulation environment for training ONLY.

# Simulates the observation space (what the CNN will provide)
# and the action space (what the EMS controller will actuate).
# Not used during deployment.

# Key changes from v1:
#   - Stochastic step size: each pulse samples from a Gaussian fitted to
#     your real EMS hand-movement data (peaks = rightward, troughs = leftward)
#   - Optional zero-response events (muscle fatigue / poor contact)
#   - 4-dim observation space:
#       [norm_error, norm_cursor_x, last_dx_norm, pulse_duration_norm]
#   - std_scale parameter: curriculum scheduler in train.py ramps this from
#     0 → 1 over training so the agent sees full variance only once it can
#     already aim under low noise
# """

# import gymnasium as gym
# import numpy as np
# from gymnasium import spaces
# import random


# # ─────────────────────────────────────────────────────────────────────────────
# # ❗ REAL VALUES: replace these 5 constants with output from ems_visualise.py
# #    - MEAN_PEAK  / STD_PEAK   → gold KDE  (rightward pulses)
# #    - MEAN_TROUGH / STD_TROUGH → blue KDE  (leftward pulses)
# #    - P_NO_RESPONSE            → fraction of pulses that produce no movement
# #      (estimate from your data: # near-zero displacements / total pulses)
# #
# #    Pulse duration: set PULSE_DURATION_MS to match your EMS timing (800 here).
# #    MAX_PLAUSIBLE_DX should be the largest single-pulse displacement you've
# #    ever observed — used only for normalisation, not clipping.
# # ─────────────────────────────────────────────────────────────────────────────
# MEAN_PEAK       = 15.0   # px — average rightward displacement per 800ms pulse
# STD_PEAK        = 6.0    # px — std of rightward displacements
# MEAN_TROUGH     = 12.0   # px — average leftward displacement per 800ms pulse
# STD_TROUGH      = 5.0    # px — std of leftward displacements
# P_NO_RESPONSE   = 0.05   # probability a pulse produces zero movement (fatigue)
# PULSE_DURATION_MS = 800  # ms — your EMS pulse width; change if you test others
# MAX_PLAUSIBLE_DX  = 60   # px — normalisation ceiling for last_dx observation
# # ─────────────────────────────────────────────────────────────────────────────


# class SimTarget:
#     """
#     Spawns a static target at a random x position.
#     Resets to a new random position on each hit.
#     """

#     def __init__(self, screen_w=1280, target_radius=30):
#         self.screen_w = screen_w
#         self.radius = target_radius
#         self.x = screen_w / 2

#     def reset(self):
#         self.x = random.uniform(self.radius * 2, self.screen_w - self.radius * 2)
#         return self.x

#     def is_hit(self, cursor_x):
#         return abs(cursor_x - self.x) < self.radius


# class AimingEnv(gym.Env):
#     """
#     Training-only simulation environment.

#     Observation space — 4 floats, mirrors what the CNN + bookkeeping will
#     provide in deployment:

#         norm_error        : (cursor_x - target_x) / screen_w  ∈ [-1,  1]
#         norm_cursor_x     : cursor_x / screen_w                ∈ [ 0,  1]
#         last_dx_norm      : last actual displacement / MAX_PLAUSIBLE_DX
#                             ∈ [-1, 1]  (negative = moved left)
#         pulse_duration_norm: pulse_duration_ms / 1000           ∈ [ 0,  1]
#                             Always 0.8 for 800ms; hook for multi-timing tests.

#     Action space:
#         0 = left
#         1 = right
#         2 = click

#     Stochastic movement:
#         Each left/right action samples dx from:
#             N(MEAN_PEAK,   std_scale * STD_PEAK)    for right
#             N(MEAN_TROUGH, std_scale * STD_TROUGH)  for left
#         std_scale is set externally by the curriculum scheduler in train.py.
#         At std_scale=0 the env is deterministic (mean only).
#         At std_scale=1 it matches your real measured variance.

#     Reward:
#         Every step : -|pixel_error| / screen_w
#         Hit        : +10.0
#         Miss click : -2.0
#         Timeout    : -5.0
#     """

#     metadata = {"render_modes": ["human"]}

#     ACTION_MAP = {
#         0: "left",
#         1: "right",
#         2: "click",
#     }

#     def __init__(
#         self,
#         screen_w=1280,
#         max_steps=300,
#         target_radius=30,
#         fps_cap=60,
#         render_mode=None,
#         # ── EMS movement distribution ─────────────────────────────────────
#         mean_peak=MEAN_PEAK,
#         std_peak=STD_PEAK,
#         mean_trough=MEAN_TROUGH,
#         std_trough=STD_TROUGH,
#         p_no_response=P_NO_RESPONSE,
#         pulse_duration_ms=PULSE_DURATION_MS,
#         # ── Curriculum knob (train.py ramps this 0 → 1) ──────────────────
#         std_scale=1.0,
#     ):
#         super().__init__()

#         self.screen_w        = screen_w
#         self.max_steps       = max_steps
#         self.target_radius   = target_radius
#         self.fps_cap         = fps_cap
#         self.render_mode     = render_mode

#         # EMS distribution parameters
#         self.mean_peak       = mean_peak
#         self.std_peak        = std_peak
#         self.mean_trough     = mean_trough
#         self.std_trough      = std_trough
#         self.p_no_response   = p_no_response
#         self.pulse_duration_ms = pulse_duration_ms

#         # Curriculum: externally updated by CurriculumCallback in train.py
#         self.std_scale = std_scale

#         # ── Observation space: 4 floats ───────────────────────────────────
#         self.observation_space = spaces.Box(
#             low  = np.array([-1.0,  0.0, -1.0,  0.0], dtype=np.float32),
#             high = np.array([ 1.0,  1.0,  1.0,  1.0], dtype=np.float32),
#         )

#         self.action_space = spaces.Discrete(3)

#         # Internal state
#         self._step_count     = 0
#         self._cursor_x       = screen_w / 2
#         self._target_x       = screen_w / 2
#         self._last_dx        = 0.0   # actual px moved last step (signed)
#         self._episode_hits   = 0
#         self._episode_misses = 0
#         self._total_stim     = 0
#         self._renderer       = None
#         self._clock          = None
#         self._font           = None
#         self._sim_target     = SimTarget(screen_w=screen_w, target_radius=target_radius)

#     # ── Observation builder ───────────────────────────────────────────────────

#     def _build_obs(self):
#         norm_error    = (self._cursor_x - self._target_x) / self.screen_w
#         norm_cursor   = self._cursor_x / self.screen_w
#         last_dx_norm  = np.clip(self._last_dx / MAX_PLAUSIBLE_DX, -1.0, 1.0)
#         pulse_norm    = self.pulse_duration_ms / 1000.0   # 0.8 for 800ms
#         return np.array(
#             [norm_error, norm_cursor, last_dx_norm, pulse_norm],
#             dtype=np.float32,
#         )

#     # ── Stochastic displacement sampler ──────────────────────────────────────

#     def _sample_dx(self, direction: str) -> float:
#         """
#         Sample the actual pixel displacement for one EMS pulse.

#         direction: "left" or "right"

#         With probability p_no_response the muscle doesn't fire (returns 0).
#         Otherwise draws from the appropriate Gaussian, clipped to [0, screen_w]
#         so we never get negative displacement magnitudes.

#         std_scale=0  → deterministic (mean only)
#         std_scale=1  → full measured variance
#         """
#         if random.random() < self.p_no_response:
#             return 0.0  # muscle fatigue / poor contact — no movement this pulse

#         if direction == "right":
#             dx = np.random.normal(
#                 loc=self.mean_peak,
#                 scale=max(self.std_scale * self.std_peak, 1e-6),
#             )
#         else:
#             dx = np.random.normal(
#                 loc=self.mean_trough,
#                 scale=max(self.std_scale * self.std_trough, 1e-6),
#             )

#         # Clamp: displacement must be non-negative (direction handled by caller)
#         return float(max(dx, 0.0))

#     # ── Gym API ───────────────────────────────────────────────────────────────

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self._step_count    = 0
#         self._cursor_x      = self.screen_w / 2
#         self._last_dx       = 0.0
#         self._episode_hits  = 0
#         self._episode_misses = 0
#         self._total_stim    = 0
#         self._target_x      = self._sim_target.reset()
#         return self._build_obs(), {}

#     def step(self, action):
#         self._step_count += 1
#         action_name = self.ACTION_MAP[int(action)]

#         actual_dx = 0.0

#         if action_name == "left":
#             dx = self._sample_dx("left")
#             self._cursor_x = max(0.0, self._cursor_x - dx)
#             actual_dx = -dx   # negative = moved left
#             self._total_stim += 1

#         elif action_name == "right":
#             dx = self._sample_dx("right")
#             self._cursor_x = min(float(self.screen_w), self._cursor_x + dx)
#             actual_dx = dx    # positive = moved right
#             self._total_stim += 1

#         self._last_dx = actual_dx

#         pixel_error = abs(self._cursor_x - self._target_x)
#         reward = -(pixel_error / self.screen_w)

#         terminated = False
#         info = {}

#         if action_name == "click":
#             if pixel_error < self.target_radius:
#                 reward += 10.0
#                 self._episode_hits += 1
#                 self._target_x = self._sim_target.reset()
#                 info["hit"] = True
#             else:
#                 reward -= 2.0
#                 self._episode_misses += 1
#                 info["miss"] = True

#         truncated = self._step_count >= self.max_steps
#         if truncated:
#             reward -= 5.0

#         info.update({
#             "pixel_error":  pixel_error,
#             "hits":         self._episode_hits,
#             "misses":       self._episode_misses,
#             "cursor_x":     self._cursor_x,
#             "target_x":     self._target_x,
#             "total_stim":   self._total_stim,
#             "actual_dx":    actual_dx,
#             "std_scale":    self.std_scale,
#         })

#         return self._build_obs(), reward, terminated, truncated, info

#     # ── Rendering ─────────────────────────────────────────────────────────────

#     def render(self):
#         if self.render_mode == "human":
#             self._render_pygame()

#     def _render_pygame(self):
#         import pygame
#         if self._renderer is None:
#             pygame.init()
#             self._renderer = pygame.display.set_mode((self.screen_w, 420))
#             pygame.display.set_caption("AimingEnv — Training Sim")
#             self._clock = pygame.time.Clock()
#             self._font  = pygame.font.SysFont("monospace", 16)

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 return

#         surf = self._renderer
#         surf.fill((10, 15, 35))

#         tx  = int(self._target_x)
#         cx  = int(self._cursor_x)
#         err = abs(cx - tx)

#         # Target
#         pygame.draw.circle(surf, (50, 180, 255),  (tx, 200), self.target_radius)
#         pygame.draw.circle(surf, (120, 220, 255), (tx, 200), 6)

#         # Cursor crosshair
#         pygame.draw.line(surf, (0, 255, 120), (cx, 180), (cx, 220), 2)
#         pygame.draw.line(surf, (0, 255, 120), (cx - 20, 200), (cx + 20, 200), 2)

#         # Accuracy bar
#         bar_w = int((1 - err / self.screen_w) * (self.screen_w - 40))
#         color = (0, 220, 100) if err < self.target_radius else (255, 100, 50)
#         pygame.draw.rect(surf, (30, 30, 50),  (20, 370, self.screen_w - 40, 14))
#         pygame.draw.rect(surf, color,         (20, 370, bar_w, 14))

#         # Curriculum variance bar (std_scale)
#         scale_w = int(self.std_scale * (self.screen_w - 40))
#         pygame.draw.rect(surf, (30, 30, 50),    (20, 390, self.screen_w - 40, 8))
#         pygame.draw.rect(surf, (200, 100, 255), (20, 390, scale_w, 8))

#         for i, line in enumerate([
#             f"Step {self._step_count}/{self.max_steps}",
#             f"Error: {err:.0f}px   last_dx: {self._last_dx:+.1f}px",
#             f"Hits: {self._episode_hits}   Misses: {self._episode_misses}",
#             f"Stim: {self._total_stim}   std_scale: {self.std_scale:.2f}",
#             f"Pulse: {self.pulse_duration_ms}ms",
#         ]):
#             surf.blit(self._font.render(line, True, (160, 180, 200)), (20, 20 + i * 22))

#         pygame.display.flip()
#         self._clock.tick(self.fps_cap)

#     def close(self):
#         if self._renderer is not None:
#             import pygame
#             pygame.quit()
#             self._renderer = None

"""
aiming_env.py — Simulation environment for training ONLY.

Simulates the observation space (what the CNN will provide)
and the action space (what the EMS controller will actuate).
Not used during deployment.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random


class SimTarget:
    """
    Spawns a static target at a random x position.
    Resets to a new random position on each hit — matching aiming.pro behaviour.
    """

    def __init__(self, screen_w=1280, target_radius=30):
        self.screen_w = screen_w
        self.radius = target_radius
        self.x = screen_w / 2
        self.reset()

    def reset(self):
        self.x = random.uniform(self.radius * 2, self.screen_w - self.radius * 2)
        return self.x

    def is_hit(self, cursor_x):
        return abs(cursor_x - self.x) < self.radius

class AimingEnv(gym.Env):
    """
    Training-only simulation environment.

    Observation space — mirrors what the CNN will provide in deployment:
        [norm_error, norm_cursor_x]
        norm_error : (cursor_x - target_x) / screen_w  ∈ [-1, 1]
        norm_cursor_x : cursor_x / screen_w  ∈ [0,  1]

    Note: target_dir removed — static targets have no velocity.

    Action space — mirrors what the EMS controller will actuate:
        0 = none
        1 = left  low
        2 = left  high
        3 = right low
        4 = right high
        5 = click

    Reward:
        Every step : -|pixel_error| / screen_w
        Hit        : +10.0
        Miss click : -2.0
        Timeout    : -5.0
        Stim cost  : -0.01 (low) / -0.03 (high) per step
    """

    metadata = {"render_modes": ["human"]}

    ACTION_MAP = {
        # 0: ("none"),
        0: ("left"),
        1: ("right"),
        2: ("click"),
    }

    def __init__(
        self,
        screen_w=1280,
        step_px=12,
        max_steps=300,
        target_radius=30,
        fps_cap=60,
        render_mode=None,
    ):
        super().__init__()

        self.screen_w = screen_w
        self.step_px = step_px
        self.max_steps = max_steps
        self.target_radius = target_radius
        self.fps_cap = fps_cap
        self.render_mode = render_mode

        # 2-float observation — matches CNN output contract
        self.observation_space = spaces.Box(
            low=np.array( [-1.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0], dtype=np.float32),
        )

        self.action_space = spaces.Discrete(3)

        self._step_count = 0
        self._cursor_x = screen_w / 2
        self._target_x = screen_w / 2
        self._episode_hits = 0
        self._episode_misses = 0
        self._total_stim = 0
        self._renderer = None
        self._sim_target = SimTarget(screen_w=screen_w, target_radius=target_radius)

    def _build_obs(self):
        error = (self._cursor_x - self._target_x) / self.screen_w
        cursor_norm = self._cursor_x / self.screen_w
        return np.array([error, cursor_norm], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._cursor_x = self.screen_w / 2
        self._episode_hits = 0
        self._episode_misses = 0
        self._total_stim = 0
        self._target_x = self._sim_target.reset()
        return self._build_obs(), {}

    def step(self, action):
        self._step_count += 1
        action_ = self.ACTION_MAP[int(action)]

        if action_ == "left":
            self._cursor_x = max(0, self._cursor_x - self.step_px)
            self._total_stim += 1
        elif action_ == "right":
            self._cursor_x = min(self.screen_w, self._cursor_x + self.step_px)
            self._total_stim += 1

        pixel_error = abs(self._cursor_x - self._target_x)
        reward = -(pixel_error / self.screen_w)

        terminated = False
        info = {}

        if action_ == "click":
            if pixel_error < self.target_radius:
                reward += 10.0
                self._episode_hits += 1
                self._target_x = self._sim_target.reset()
                info["hit"] = True
            else:
                reward -= 2.0
                self._episode_misses += 1
                info["miss"] = True

        truncated = self._step_count >= self.max_steps
        if truncated:
            reward -= 5.0

        info.update({
            "pixel_error": pixel_error,
            "hits":        self._episode_hits,
            "misses":      self._episode_misses,
            "cursor_x":    self._cursor_x,
            "target_x":    self._target_x,
            "total_stim":  self._total_stim,
        })

        return self._build_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_pygame()

    def _render_pygame(self):
        import pygame
        if self._renderer is None:
            pygame.init()
            self._renderer = pygame.display.set_mode((self.screen_w, 400))
            pygame.display.set_caption("AimingEnv — Training Sim")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("monospace", 16)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        surf = self._renderer
        surf.fill((10, 15, 35))

        tx = int(self._target_x)
        cx = int(self._cursor_x)
        pixel_error = abs(cx - tx)

        pygame.draw.circle(surf, (50, 180, 255), (tx, 200), self.target_radius)
        pygame.draw.circle(surf, (120, 220, 255), (tx, 200), 6)
        pygame.draw.line(surf, (0, 255, 120), (cx, 180), (cx, 220), 2)
        pygame.draw.line(surf, (0, 255, 120), (cx - 20, 200), (cx + 20, 200), 2)

        bar_w = int((1 - pixel_error / self.screen_w) * (self.screen_w - 40))
        color = (0, 220, 100) if pixel_error < self.target_radius else (255, 100, 50)
        pygame.draw.rect(surf, (30, 30, 50), (20, 360, self.screen_w - 40, 14))
        pygame.draw.rect(surf, color, (20, 360, bar_w, 14))

        for i, line in enumerate([
            f"Step {self._step_count}/{self.max_steps}",
            f"Error: {pixel_error:.0f}px",
            f"Hits: {self._episode_hits}  Misses: {self._episode_misses}",
            f"Total stim: {self._total_stim}",
        ]):
            surf.blit(self._font.render(line, True, (160, 180, 200)), (20, 20 + i * 22))

        pygame.display.flip()
        self._clock.tick(self.fps_cap)

    def close(self):
        if self._renderer is not None:
            import pygame
            pygame.quit()
            self._renderer = None