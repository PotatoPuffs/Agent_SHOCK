"""
interfacing.py — Integration contracts for the Agent SHOCK pipeline.

This file defines:
  - What the RL agent expects FROM the CNN observer
  - What the RL agent sends TO the EMS controller

Both teammates should implement to these interfaces.
The RL agent only depends on this file — not on their actual implementations.
"""

from abc import ABC, abstractmethod
import numpy as np


# ── What the RL agent expects from the CNN ──────────────────────────

class BaseCNNObserver(ABC):
    """
    The CNN teammate implements this.
    The RL agent calls build_obs() every frame.
    """

    @abstractmethod
    def build_obs(self, cursor_x: float) -> tuple[np.ndarray, float]:
        """
        Capture the game frame and return the RL observation.

        Args:
            cursor_x : current virtual cursor position in pixels

        Returns:
            obs      : np.ndarray shape (3,) dtype float32
                         [0] norm_error      — (cursor_x - target_x) / screen_w  ∈ [-1, 1]
                         [1] norm_cursor_x   — cursor_x / screen_w                ∈ [0,  1]
                         [2] target_dir      — sign of target movement             ∈ {-1, 0, 1}
            target_x : float — raw target x position in pixels (for env.update_target_x)
        """
        pass


# ── What the RL agent sends to the EMS ───────────────────────────────

class BaseEMSController(ABC):
    """
    The EMS teammate implements this.
    The RL agent calls send_action() every frame.
    """

    @abstractmethod
    def send_action(self, direction: str | None, intensity: str) -> None:
        """
        Send a stimulation command to the Arduino.

        Args:
            direction : 'left' | 'right' | None
            intensity : 'none' | 'low' | 'high' | 'click'

        Expected behaviour per intensity:
            'none'  — send nothing
            'low'   — short pulse  (~30ms)
            'high'  — long pulse   (~80ms)
            'click' — click event
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Called after each click attempt to reset the stimulation budget."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up serial connection on shutdown."""
        pass