"""
simulators.py — Drop-in simulated implementations of CNN and EMS interfaces.

Swap in/out via CLI flags in run.py — no import editing needed.

SimulatedCNNObserver:
    - Spawns a STATIC target (matching aiming.pro drill 52502 behaviour)
    - Tracks cursor_x internally from accumulated EMS displacements
    - Returns the correct 4-float obs matching the training contract

SimulatedEMSController:
    - Draws displacement from a Gaussian fitted to real EMS data
    - Returns actual_dx so the loop can update last_dx
    - Prints what it would send to the Arduino

❗ REAL VALUES: replace MEAN_PEAK / STD_PEAK / MEAN_TROUGH / STD_TROUGH
   with values from ems_visualise.py once you have real hand-movement data.
"""

import numpy as np
import random
from integration.interfacing import (BaseCNNObserver, BaseEMSController, 
                                     SCREEN_W, TARGET_RADIUS, PULSE_DURATION_MS, MAX_DX)

# ── EMS displacement distribution ────────────────────────────────────────────
# ❗ REAL VALUES: replace these with output from ems_visualise.py
MEAN_PEAK    = 15.0   # px — average rightward displacement per pulse
STD_PEAK     = 6.0    # px — std of rightward displacements
MEAN_TROUGH  = 12.0   # px — average leftward displacement per pulse
STD_TROUGH   = 5.0    # px — std of leftward displacements
P_NO_RESP    = 0.05   # probability a pulse produces zero movement (fatigue)
# ─────────────────────────────────────────────────────────────────────────────


class SimulatedCNNObserver(BaseCNNObserver):
    """
    Fake CNN observer — no screen capture.

    Simulates a STATIC target (spawned once per episode at a random x),
    matching aiming.pro drill 52502 where targets don't move.

    cursor_x is tracked internally and updated by the EMS simulator
    via update_cursor() after each pulse, so the obs reflects what
    the hand actually did rather than what was intended.
    """

    def __init__(self, screen_w: int = SCREEN_W):
        self.screen_w  = screen_w
        self._cursor_x = screen_w / 2.0
        self._target_x = self._spawn_target()

    def _spawn_target(self) -> float:
        margin = TARGET_RADIUS * 2
        return random.uniform(margin, self.screen_w - margin)

    def update_cursor(self, new_cursor_x: float) -> None:
        """Called by SimulatedEMSController after each pulse to move cursor."""
        self._cursor_x = float(np.clip(new_cursor_x, 0, self.screen_w))

    def reset_target(self) -> None:
        """Spawn a new static target — call after each successful hit."""
        self._target_x = self._spawn_target()

    def get_state(
        self,
        last_dx: float,
        pulse_duration_ms: float,
    ) -> tuple[np.ndarray, float, float]:
        norm_error    = (self._cursor_x - self._target_x) / self.screen_w
        norm_cursor   = self._cursor_x / self.screen_w
        last_dx_norm  = float(np.clip(last_dx / MAX_DX, -1.0, 1.0))
        pulse_norm    = pulse_duration_ms / 1000.0

        obs = np.array(
            [norm_error, norm_cursor, last_dx_norm, pulse_norm],
            dtype=np.float32,
        )
        return obs, self._target_x, self._cursor_x


class SimulatedEMSController(BaseEMSController):
    """
    Fake EMS — simulates stochastic hand movement instead of firing relays.

    Each pulse samples displacement from a Gaussian fitted to real EMS data
    so that the simulated deployment loop behaves like the real one.
    Returns actual_dx so run.py can pass it back as last_dx next frame.
    """

    def __init__(self, observer: SimulatedCNNObserver, std_scale: float = 1.0):
        """
        Args:
            observer  : SimulatedCNNObserver — updated after each pulse so
                        the obs reflects the simulated hand position.
            std_scale : 0.0 = deterministic mean movement, 1.0 = full variance.
                        Curriculum scheduler in train can ramp this externally.
        """
        self.observer   = observer
        self.std_scale  = std_scale
        self._pulse_count = 0
        print("[SimEMS] Simulated EMS active — no hardware connected.")

    def _sample_dx(self, direction: str) -> float:
        """Draw a stochastic displacement for one pulse."""
        if random.random() < P_NO_RESP:
            return 0.0  # muscle fatigue / poor contact

        if direction == "right":
            dx = np.random.normal(MEAN_PEAK,   max(self.std_scale * STD_PEAK,   1e-6))
        else:
            dx = np.random.normal(MEAN_TROUGH, max(self.std_scale * STD_TROUGH, 1e-6))

        return float(max(dx, 0.0))  # magnitude is always non-negative

    def send_action(self, action: str) -> float:
        """
        Simulate the EMS pulse and update the observer's cursor position.

        Returns:
            actual_dx : signed pixel displacement (positive=right, negative=left).
                        For 'click' and 'none', returns 0.0.
                        run.py stores this and passes it back as last_dx.
        """
        actual_dx = 0.0

        if action == "right":
            dx = self._sample_dx("right")
            new_x = self.observer._cursor_x + dx
            self.observer.update_cursor(new_x)
            actual_dx = dx
            self._pulse_count += 1
            print(f"[SimEMS] → RIGHT  dx={dx:+.1f}px  "
                  f"cursor={self.observer._cursor_x:.0f}px  "
                  f"(pulse #{self._pulse_count})")

        elif action == "left":
            dx = self._sample_dx("left")
            new_x = self.observer._cursor_x - dx
            self.observer.update_cursor(new_x)
            actual_dx = -dx
            self._pulse_count += 1
            print(f"[SimEMS] → LEFT   dx={dx:+.1f}px  "
                  f"cursor={self.observer._cursor_x:.0f}px  "
                  f"(pulse #{self._pulse_count})")

        elif action == "click":
            self._pulse_count += 1
            print(f"[SimEMS] → CLICK (momentary pulse) (pulse #{self._pulse_count})")

        elif action == "none":
            print(f"[SimEMS] → NONE (all relays open)")

        return actual_dx

    def close(self) -> None:
        print(f"[SimEMS] Closed. Total pulses sent: {self._pulse_count}")



# """
# stubs.py — Simulated implementations of CNN and EMS interfaces.

# Use these to develop and test the RL agent in isolation.
# Swap them out for real implementations when teammates are ready.

# Usage in run_agent.py:
#     from stubs import SimulatedCNNObserver, SimulatedEMSController
#     # replace with real classes when ready:
#     # from perception.cnn_observer import CNNObserver
#     # from hardware.ems_controller import EMSController
# """

# import numpy as np
# import random
# from interfacing import BaseCNNObserver, BaseEMSController


# class SimulatedCNNObserver(BaseCNNObserver):
#     """
#     Fake CNN — generates a plausible observation without any screen capture.
#     Simulates a target drifting slowly across the screen.
#     Use this to test the RL loop before the CNN is ready.
#     """

#     def __init__(self, screen_w: int = 1280):
#         self.screen_w = screen_w
#         self._target_x = screen_w / 2
#         self._vx = random.choice([-1, 1]) * 3.0

#     def build_obs(self, cursor_x: float) -> tuple[np.ndarray, float]:
#         # Simulate slow target drift
#         self._target_x += self._vx
#         if self._target_x < 50 or self._target_x > self.screen_w - 50:
#             self._vx *= -1
#         self._target_x = np.clip(self._target_x, 0, self.screen_w)

#         error       = (cursor_x - self._target_x) / self.screen_w
#         cursor_norm = cursor_x / self.screen_w
#         target_dir  = float(np.sign(self._vx))

#         obs = np.array([error, cursor_norm, target_dir], dtype=np.float32)
#         return obs, self._target_x


# class SimulatedEMSController(BaseEMSController):
#     """
#     Fake EMS — just prints what would be sent to the Arduino.
#     Use this to verify the agent is outputting correct actions
#     before the EMS hardware is ready.
#     """

#     def __init__(self):
#         self._stim_count = 0
#         print("[SimulatedEMS] Simulated EMS controller active — no hardware connected")

#     def send_action(self, direction: str | None, intensity: str) -> None:
#         if intensity in ("none", "click") or direction is None:
#             return
#         self._stim_count += 1
#         print(f"[SimulatedEMS] → direction={direction}  intensity={intensity}  "
#               f"(total pulses this episode: {self._stim_count})")

#     def reset(self) -> None:
#         self._stim_count = 0

#     def close(self) -> None:
#         print(f"[SimulatedEMS] Closed. Total pulses sent: {self._stim_count}")