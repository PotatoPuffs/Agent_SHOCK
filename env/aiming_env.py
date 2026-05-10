"""
AimingEnv — Custom OpenAI Gym environment for aiming.pro

Modes:
  'sim'    — fully virtual, no screen capture, no real mouse. Fast. Use for training.
  'screen' — captures live browser screen, moves a virtual internal cursor. No real mouse.
  'real'   — screen capture + sends serial commands to MCU for EMS. Use for deployment.

The agent NEVER moves the real OS mouse in sim or screen mode.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
import random


# ─── Sim-mode target dynamics ────────────────────────────────────────────────

class SimTarget:
    """Simulates a moving target in the aiming.pro style (horizontal drift)."""

    def __init__(self, screen_w=1280, speed_range=(1, 4), target_radius=30):
        self.screen_w = screen_w
        self.speed_range = speed_range
        self.radius = target_radius
        self.reset()

    def reset(self):
        self.x = random.uniform(self.radius, self.screen_w - self.radius)
        self.vx = random.choice([-1, 1]) * random.uniform(*self.speed_range)
        return self.x

    def step(self):
        self.x += self.vx
        # Bounce off walls
        if self.x < self.radius or self.x > self.screen_w - self.radius:
            self.vx *= -1
            self.x = np.clip(self.x, self.radius, self.screen_w - self.radius)
        return self.x

    def is_hit(self, cursor_x):
        return abs(cursor_x - self.x) < self.radius


# ─── Main Gym Environment ─────────────────────────────────────────────────────

class AimingEnv(gym.Env):
    """
    Observation space:
        [pixel_error_normalised, cursor_x_normalised, target_vx_sign]
        All values in [-1, 1]

    Action space (Discrete 3):
        0 = move left   (cursor_x -= step_px)
        1 = move right  (cursor_x += step_px)
        2 = click

    Reward:
        Each step:   -|pixel_error| / screen_w   (penalise being off-target)
        On hit:      +10.0                        (click when aligned)
        On miss:     -2.0                         (click when not aligned)
        On timeout:  -5.0                         (never clicked in time)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        mode="sim",              # 'sim' | 'screen' | 'real'
        screen_w=1280,
        screen_h=720,
        step_px=8,               # pixels cursor moves per action
        max_steps=300,           # steps before episode ends
        target_radius=30,
        fps_cap=60,
        render_mode=None,
        serial_port=None,        # for 'real' mode, e.g. '/dev/ttyUSB0'
    ):
        super().__init__()

        self.mode = mode
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.step_px = step_px
        self.max_steps = max_steps
        self.target_radius = target_radius
        self.fps_cap = fps_cap
        self.render_mode = render_mode
        self.serial_port = serial_port

        # Observation: [norm_error, norm_cursor_x, target_dir]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0,  1.0], dtype=np.float32),
        )

        # Actions: left, right, click
        self.action_space = spaces.Discrete(3)

        self._step_count = 0
        self._cursor_x = screen_w / 2
        self._target = SimTarget(screen_w, target_radius=target_radius)
        self._episode_hits = 0
        self._episode_misses = 0
        self._last_hit_step = 0

        # Screen mode setup
        if mode in ("screen", "real"):
            self._setup_screen_capture()

        # Real mode setup
        if mode == "real":
            self._setup_serial(serial_port)

        # Renderer
        self._renderer = None

    # ── Setup helpers ──────────────────────────────────────────────────────────

    def _setup_screen_capture(self):
        try:
            import mss
            self._sct = mss.mss()
            # Define capture region — update to match your browser window
            self._monitor = {
                "top": 130, "left": 0,
                "width": self.screen_w, "height": self.screen_h
            }
        except ImportError:
            raise ImportError("pip install mss  — required for screen/real mode")

    def _setup_serial(self, port):
        if port is None:
            raise ValueError("serial_port required for real mode (e.g. '/dev/ttyUSB0')")
        try:
            import serial
            self._ser = serial.Serial(port, baudrate=115200, timeout=1)
            time.sleep(2)  # MCU reset
        except ImportError:
            raise ImportError("pip install pyserial  — required for real mode")

    # ── Observation helpers ────────────────────────────────────────────────────

    def _get_target_x_from_screen(self):
        """Capture screen and return target x-pixel via YOLO or colour blob."""
        import numpy as np
        from PIL import Image

        img = self._sct.grab(self._monitor)
        frame = np.array(img)[:, :, :3]  # drop alpha

        # Try YOLO first; fall back to colour blob
        if hasattr(self, '_yolo'):
            return self._detect_yolo(frame)
        else:
            return self._detect_blob(frame)

    def _detect_blob(self, frame):
        """Simple HSV blue-blob detector for aiming.pro's cyan targets."""
        import cv2
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        # aiming.pro targets are bright cyan/blue
        mask = cv2.inRange(hsv, np.array([85, 150, 150]), np.array([130, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # Return x-centre of largest contour
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        return int(M["m10"] / M["m00"])

    def _detect_yolo(self, frame):
        results = self._yolo(frame, verbose=False)
        if len(results[0].boxes) == 0:
            return None
        box = results[0].boxes[0].xyxy[0].cpu().numpy()
        return int((box[0] + box[2]) / 2)

    def load_yolo(self, model_path):
        """Call this to swap blob detection for a trained YOLO model."""
        from ultralytics import YOLO
        self._yolo = YOLO(model_path)

    def _get_obs(self):
        target_x = self._target_x
        error = (self._cursor_x - target_x) / self.screen_w   # normalised [-1,1]
        cursor_norm = self._cursor_x / self.screen_w           # normalised [0,1]
        dir_sign = np.sign(self._target.vx)                    # -1 or +1
        return np.array([error, cursor_norm, dir_sign], dtype=np.float32)

    # ── Core Gym API ───────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._cursor_x = self.screen_w / 2
        self._episode_hits = 0
        self._episode_misses = 0
        self._target_x = self._target.reset()
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1

        # 1. Move virtual cursor
        if action == 0:
            self._cursor_x = max(0, self._cursor_x - self.step_px)
        elif action == 1:
            self._cursor_x = min(self.screen_w, self._cursor_x + self.step_px)

        # 2. Advance target
        if self.mode == "sim":
            self._target_x = self._target.step()
        else:
            detected = self._get_target_x_from_screen()
            self._target_x = detected if detected is not None else self._target_x

        # 3. Compute reward
        pixel_error = abs(self._cursor_x - self._target_x)
        reward = -(pixel_error / self.screen_w)  # step penalty

        terminated = False
        info = {}

        if action == 2:  # click
            if self._target.is_hit(self._cursor_x) or pixel_error < self.target_radius:
                reward += 10.0
                self._episode_hits += 1
                info["hit"] = True
                # Reset target position on hit (like the game does)
                self._target_x = self._target.reset()
            else:
                reward += -2.0
                self._episode_misses += 1
                info["miss"] = True

        # 4. Real mode: send EMS command
        if self.mode == "real" and action in (0, 1):
            channel = 0 if action == 0 else 1
            self._send_ems(channel, duration_ms=50)

        # 5. Timeout
        truncated = self._step_count >= self.max_steps
        if truncated:
            reward -= 5.0

        info.update({
            "pixel_error": pixel_error,
            "hits": self._episode_hits,
            "misses": self._episode_misses,
            "cursor_x": self._cursor_x,
            "target_x": self._target_x,
        })

        return self._get_obs(), reward, terminated, truncated, info

    # ── EMS serial output ──────────────────────────────────────────────────────

    def _send_ems(self, channel, duration_ms):
        """
        Send EMS command to MCU over serial.
        Protocol: 'C<channel>D<duration_ms>\\n'
        e.g.  'C0D50\\n' = channel 0, 50ms pulse
        """
        cmd = f"C{channel}D{duration_ms}\n".encode()
        self._ser.write(cmd)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "human":
            self._render_pygame()

    def _render_pygame(self):
        import pygame
        if self._renderer is None:
            pygame.init()
            self._renderer = pygame.display.set_mode((self.screen_w, 400))
            pygame.display.set_caption("AimingEnv — Virtual Cursor")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("monospace", 16)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        surf = self._renderer
        surf.fill((10, 15, 35))

        # Grid lines
        for i in range(0, self.screen_w, 80):
            pygame.draw.line(surf, (25, 35, 70), (i, 0), (i, 400), 1)

        # Target
        tx = int(self._target_x)
        pygame.draw.circle(surf, (50, 180, 255), (tx, 200), self.target_radius)
        pygame.draw.circle(surf, (120, 220, 255), (tx, 200), 6)

        # Crosshair
        cx = int(self._cursor_x)
        pygame.draw.line(surf, (0, 255, 120), (cx, 180), (cx, 220), 2)
        pygame.draw.line(surf, (0, 255, 120), (cx - 20, 200), (cx + 20, 200), 2)

        # Error bar
        error_px = abs(cx - tx)
        bar_w = int((1 - error_px / self.screen_w) * (self.screen_w - 40))
        pygame.draw.rect(surf, (30, 30, 50), (20, 360, self.screen_w - 40, 14))
        color = (0, 220, 100) if error_px < self.target_radius else (255, 100, 50)
        pygame.draw.rect(surf, color, (20, 360, bar_w, 14))

        # Stats
        lines = [
            f"Step {self._step_count}/{self.max_steps}",
            f"Error: {error_px:.0f}px",
            f"Hits: {self._episode_hits}  Misses: {self._episode_misses}",
        ]
        for i, l in enumerate(lines):
            surf.blit(self._font.render(l, True, (160, 180, 200)), (20, 20 + i * 22))

        pygame.display.flip()
        self._clock.tick(self.fps_cap)

    def close(self):
        if self._renderer is not None:
            import pygame
            pygame.quit()
            self._renderer = None
        if self.mode == "real" and hasattr(self, "_ser"):
            self._ser.close()
