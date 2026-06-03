"""
aiming_env.py — Gymnasium training environment (simulation only).

Used exclusively during `python run.py --mode train`.
Not used during deployment — the live game is the environment then.

Observation space matches the CNN contract in interfacing.py exactly:
    [norm_error, norm_cursor_x, last_dx_norm, pulse_dur_norm]

Action space:
    0 = left
    1 = right
    2 = click
    3 = none

Stochastic movement:
    Each pulse samples dx ~ N(mean, std_scale * std) so the agent learns
    to tolerate real EMS variance. std_scale is ramped 0->1 by the
    CurriculumCallback in run.py.

REAL VALUES: replace the five constants below with output from
   ems_visualise.py once you have real hand-movement data.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random

from integration.interfacing import (
    SCREEN_W, TARGET_RADIUS, PULSE_DURATION_MS, MAX_DX, OBS_SIZE,
)

# EMS displacement distribution
# REAL VALUES: paste values from ems_visualise.py output here
MEAN_PEAK    = 15.0   # px - average rightward displacement per pulse
STD_PEAK     = 6.0    # px - std of rightward displacements
MEAN_TROUGH  = 12.0   # px - average leftward displacement per pulse
STD_TROUGH   = 5.0    # px - std of leftward displacements
P_NO_RESP    = 0.05   # probability a pulse produces zero movement


class SimTarget:
    """Static target - spawns at a random x, resets on each hit."""

    def __init__(self, screen_w: int = SCREEN_W, radius: int = TARGET_RADIUS):
        self.screen_w = screen_w
        self.radius   = radius
        self.x        = screen_w / 2.0

    def reset(self) -> float:
        margin = self.radius * 2
        self.x = random.uniform(margin, self.screen_w - margin)
        return self.x


class AimingEnv(gym.Env):
    """
    Training-only simulation environment.

    The 4-float observation vector is identical to what BaseCNNObserver.get_state()
    returns in deployment, so the trained policy transfers directly.
    """

    metadata   = {"render_modes": ["human"]}
    ACTION_MAP = {0: "left", 1: "right", 2: "click"}

    def __init__(
        self,
        screen_w: int          = SCREEN_W,
        max_steps: int         = 300,
        target_radius: int     = TARGET_RADIUS,
        pulse_duration_ms: float = PULSE_DURATION_MS,
        fps_cap: int           = 60,
        render_mode            = None,
        std_scale: float       = 0.0,
        mean_peak: float       = MEAN_PEAK,
        std_peak: float        = STD_PEAK,
        mean_trough: float     = MEAN_TROUGH,
        std_trough: float      = STD_TROUGH,
        p_no_resp: float       = P_NO_RESP,
    ):
        super().__init__()

        self.screen_w          = screen_w
        self.max_steps         = max_steps
        self.target_radius     = target_radius
        self.pulse_duration_ms = pulse_duration_ms
        self.fps_cap           = fps_cap
        self.render_mode       = render_mode
        self.std_scale         = std_scale
        self.mean_peak         = mean_peak
        self.std_peak          = std_peak
        self.mean_trough       = mean_trough
        self.std_trough        = std_trough
        self.p_no_resp         = p_no_resp

        # Observation space: 4 floats, matching interfacing.py contract
        self.observation_space = spaces.Box(
            low  = np.array([-1.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high = np.array([ 1.0, 1.0,  1.0, 1.0], dtype=np.float32),
        )
        assert self.observation_space.shape[0] == OBS_SIZE, \
            f"Obs size mismatch: env={self.observation_space.shape[0]} contract={OBS_SIZE}"

        self.action_space = spaces.Discrete(3)

        self._step_count     = 0
        self._cursor_x       = float(screen_w) / 2.0
        self._target_x       = float(screen_w) / 2.0
        self._last_dx        = 0.0
        self._episode_hits   = 0
        self._episode_misses = 0
        self._total_stim     = 0
        self._sim_target     = SimTarget(screen_w=screen_w, radius=target_radius)
        self._renderer       = None
        self._clock          = None
        self._font           = None

    def _build_obs(self) -> np.ndarray:
        norm_error   = (self._cursor_x - self._target_x) / self.screen_w
        norm_cursor  = self._cursor_x / self.screen_w
        last_dx_norm = float(np.clip(self._last_dx / MAX_DX, -1.0, 1.0))
        pulse_norm   = self.pulse_duration_ms / 1000.0
        return np.array(
            [norm_error, norm_cursor, last_dx_norm, pulse_norm],
            dtype=np.float32,
        )

    def _sample_dx(self, direction: str) -> float:
        if random.random() < self.p_no_resp:
            return 0.0
        if direction == "right":
            dx = np.random.normal(
                self.mean_peak,
                max(self.std_scale * self.std_peak, 1e-6),
            )
        else:
            dx = np.random.normal(
                self.mean_trough,
                max(self.std_scale * self.std_trough, 1e-6),
            )
        return float(max(dx, 0.0))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count     = 0
        self._cursor_x       = float(self.screen_w) / 2.0
        self._last_dx        = 0.0
        self._episode_hits   = 0
        self._episode_misses = 0
        self._total_stim     = 0
        self._target_x       = self._sim_target.reset()
        return self._build_obs(), {}

    def step(self, action: int):
        self._step_count += 1
        action_name = self.ACTION_MAP[int(action)]
        actual_dx   = 0.0

        if action_name == "left":
            dx             = self._sample_dx("left")
            self._cursor_x = max(0.0, self._cursor_x - dx)
            actual_dx      = -dx
            self._total_stim += 1
        elif action_name == "right":
            dx             = self._sample_dx("right")
            self._cursor_x = min(float(self.screen_w), self._cursor_x + dx)
            actual_dx      = dx
            self._total_stim += 1

        self._last_dx = actual_dx
        pixel_error   = abs(self._cursor_x - self._target_x)
        reward        = -(pixel_error / self.screen_w)
        terminated    = False
        info          = {}

        if action_name == "click":
            if pixel_error < self.target_radius:
                reward             += 10.0
                self._episode_hits += 1
                self._target_x      = self._sim_target.reset()
                info["hit"]         = True
            else:
                reward               -= 2.0
                self._episode_misses += 1
                info["miss"]          = True

        truncated = self._step_count >= self.max_steps
        if truncated:
            reward -= 5.0

        info.update({
            "pixel_error": pixel_error,
            "hits":        self._episode_hits,
            "misses":      self._episode_misses,
            "cursor_x":    self._cursor_x,
            "target_x":    self._target_x,
            "actual_dx":   actual_dx,
            "std_scale":   self.std_scale,
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
            self._renderer = pygame.display.set_mode((self.screen_w, 430))
            pygame.display.set_caption("AimingEnv - Training Sim")
            self._clock = pygame.time.Clock()
            self._font  = pygame.font.SysFont("monospace", 16)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        surf = self._renderer
        surf.fill((10, 15, 35))
        tx  = int(self._target_x)
        cx  = int(self._cursor_x)
        err = abs(cx - tx)

        pygame.draw.circle(surf, (50, 180, 255),  (tx, 200), self.target_radius)
        pygame.draw.circle(surf, (120, 220, 255), (tx, 200), 6)
        pygame.draw.line(surf, (0, 255, 120), (cx, 178), (cx, 222), 2)
        pygame.draw.line(surf, (0, 255, 120), (cx - 22, 200), (cx + 22, 200), 2)

        bar_w = int((1 - err / self.screen_w) * (self.screen_w - 40))
        color = (0, 220, 100) if err < self.target_radius else (255, 100, 50)
        pygame.draw.rect(surf, (30, 30, 50),   (20, 370, self.screen_w - 40, 14))
        pygame.draw.rect(surf, color,           (20, 370, bar_w, 14))

        scale_w = int(self.std_scale * (self.screen_w - 40))
        pygame.draw.rect(surf, (30, 30, 50),   (20, 392, self.screen_w - 40, 8))
        pygame.draw.rect(surf, (180, 80, 255), (20, 392, scale_w, 8))

        for i, line in enumerate([
            f"Step {self._step_count}/{self.max_steps}   std_scale={self.std_scale:.2f}",
            f"Error: {err:.0f}px   last_dx: {self._last_dx:+.1f}px",
            f"Hits: {self._episode_hits}   Misses: {self._episode_misses}",
            f"Stim: {self._total_stim}   Pulse: {self.pulse_duration_ms}ms",
        ]):
            surf.blit(self._font.render(line, True, (160, 180, 200)), (20, 20 + i * 22))

        pygame.display.flip()
        self._clock.tick(self.fps_cap)

    def close(self):
        if self._renderer is not None:
            import pygame
            pygame.quit()
            self._renderer = None