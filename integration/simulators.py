"""
stubs.py — Simulated implementations of CNN and EMS interfaces.

Use these to develop and test the RL agent in isolation.
Swap them out for real implementations when teammates are ready.

Usage in run_agent.py:
    from stubs import SimulatedCNNObserver, SimulatedEMSController
    # replace with real classes when ready:
    # from perception.cnn_observer import CNNObserver
    # from hardware.ems_controller import EMSController
"""

import numpy as np
import random
from interfacing import BaseCNNObserver, BaseEMSController


class SimulatedCNNObserver(BaseCNNObserver):
    """
    Fake CNN — generates a plausible observation without any screen capture.
    Simulates a target drifting slowly across the screen.
    Use this to test the RL loop before the CNN is ready.
    """

    def __init__(self, screen_w: int = 1280):
        self.screen_w = screen_w
        self._target_x = screen_w / 2
        self._vx = random.choice([-1, 1]) * 3.0

    def build_obs(self, cursor_x: float) -> tuple[np.ndarray, float]:
        # Simulate slow target drift
        self._target_x += self._vx
        if self._target_x < 50 or self._target_x > self.screen_w - 50:
            self._vx *= -1
        self._target_x = np.clip(self._target_x, 0, self.screen_w)

        error       = (cursor_x - self._target_x) / self.screen_w
        cursor_norm = cursor_x / self.screen_w
        target_dir  = float(np.sign(self._vx))

        obs = np.array([error, cursor_norm, target_dir], dtype=np.float32)
        return obs, self._target_x


class SimulatedEMSController(BaseEMSController):
    """
    Fake EMS — just prints what would be sent to the Arduino.
    Use this to verify the agent is outputting correct actions
    before the EMS hardware is ready.
    """

    def __init__(self):
        self._stim_count = 0
        print("[SimulatedEMS] Simulated EMS controller active — no hardware connected")

    def send_action(self, direction: str | None, intensity: str) -> None:
        if intensity in ("none", "click") or direction is None:
            return
        self._stim_count += 1
        print(f"[SimulatedEMS] → direction={direction}  intensity={intensity}  "
              f"(total pulses this episode: {self._stim_count})")

    def reset(self) -> None:
        self._stim_count = 0

    def close(self) -> None:
        print(f"[SimulatedEMS] Closed. Total pulses sent: {self._stim_count}")