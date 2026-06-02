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