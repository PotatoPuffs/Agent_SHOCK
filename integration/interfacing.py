"""
interfacing.py — Integration contracts for the Agent SHOCK pipeline.

Defines exactly what flows between each component:
  - CNN observer  →  RL agent   (observation vector + positions)
  - RL agent      →  EMS controller  (action string)

All teammates implement against these base classes.
The RL agent depends ONLY on this file — not on any concrete implementation.

Observation vector contract (4 floats, dtype float32):
    [0] norm_error       — (cursor_x - target_x) / screen_w   ∈ [-1,  1]
    [1] norm_cursor_x    — cursor_x / screen_w                 ∈ [ 0,  1]
    [2] last_dx_norm     — last actual displacement / MAX_DX   ∈ [-1,  1]
    [3] pulse_dur_norm   — pulse_duration_ms / 1000            ∈ [ 0,  1]

Action contract (single string):
    'left'   — close left EMS relay (stays closed until next command)
    'right'  — close right EMS relay (stays closed until next command)
    'click'  — momentary pulse on click relay (~50-100ms)
    'none'   — open all relays (stop stimulation)
"""

from abc import ABC, abstractmethod
import numpy as np

# ── Shared constants — import these everywhere to stay in sync ────────────────

SCREEN_W          = 1280       # px — game capture width
TARGET_RADIUS     = 30         # px — hit threshold
PULSE_DURATION_MS = 200        # ms — fixed relay-close duration per pulse
MAX_DX            = 60         # px — normalisation ceiling for last_dx obs
OBS_SIZE          = 4          # length of the observation vector

# ── CNN → Agent ───────────────────────────────────────────────────────────────

class BaseCNNObserver(ABC):
    """
    Implemented by the CNN teammate.
    Called every frame by the agent loop to get the current game state.
    """

    @abstractmethod
    def get_state(
        self,
        last_dx: float,
        pulse_duration_ms: float,
    ) -> tuple[np.ndarray, float, float]:
        """
        Capture the current game frame and return the full game state.

        Args:
            last_dx           : signed pixel displacement from the previous
                                pulse (positive = moved right, negative = left).
                                Used to build last_dx_norm observation element.
            pulse_duration_ms : duration of the last pulse in ms.
                                Used to build pulse_dur_norm observation element.

        Returns:
            obs       : np.ndarray shape (4,) dtype float32
                          [0] norm_error     = (cursor_x - target_x) / SCREEN_W
                          [1] norm_cursor_x  = cursor_x / SCREEN_W
                          [2] last_dx_norm   = clip(last_dx / MAX_DX, -1, 1)
                          [3] pulse_dur_norm = pulse_duration_ms / 1000
            target_x  : float — raw target x position in pixels
            cursor_x  : float — raw cursor x position in pixels
                        (CNN detects both the target circle and the crosshair)
        """
        ...


# ── Agent → EMS ───────────────────────────────────────────────────────────────

class BaseEMSController(ABC):
    """
    Implemented by the EMS/Arduino teammate.
    Called every frame by the agent loop with the chosen action.
    """

    @abstractmethod
    def send_action(self, action: str) -> None:
        """
        Send a command to the hardware (fire-and-forget, non-blocking).

        Args:
            action : 'left'  — close left relay (stays closed until next command)
                     'right' — close right relay (stays closed until next command)
                     'click' — momentary pulse on click relay
                     'none'  — open all relays (stop stimulation)

        Returns immediately. Relay states are maintained until a different
        command is sent. The CNN observer detects the game state change and
        provides feedback to the RL agent.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release serial port / GPIO on shutdown. Always called in finally."""
        ...



# """
# interfacing.py — Integration contracts for the Agent SHOCK pipeline.

# This file defines:
#   - What the RL agent expects FROM the CNN observer
#   - What the RL agent sends TO the EMS controller

# Both teammates should implement to these interfaces.
# The RL agent only depends on this file — not on their actual implementations.
# """

# from abc import ABC, abstractmethod
# import numpy as np


# # ── What the RL agent expects from the CNN ──────────────────────────

# class BaseCNNObserver(ABC):
#     """
#     The CNN teammate implements this.
#     The RL agent calls build_obs() every frame.
#     """

#     @abstractmethod
#     def build_obs(self, cursor_x: float) -> tuple[np.ndarray, float]:
#         """
#         Capture the game frame and return the RL observation.

#         Args:
#             cursor_x : current virtual cursor position in pixels

#         Returns:
#             obs      : np.ndarray shape (3,) dtype float32
#                          [0] norm_error      — (cursor_x - target_x) / screen_w  ∈ [-1, 1]
#                          [1] norm_cursor_x   — cursor_x / screen_w                ∈ [0,  1]
#                          [2] target_dir      — sign of target movement             ∈ {-1, 0, 1}
#             target_x : float — raw target x position in pixels (for env.update_target_x)
#         """
#         pass


# # ── What the RL agent sends to the EMS ───────────────────────────────

# class BaseEMSController(ABC):
#     """
#     The EMS teammate implements this.
#     The RL agent calls send_action() every frame.
#     """

#     @abstractmethod
#     def send_action(self, direction: str | None, intensity: str) -> None:
#         """
#         Send a stimulation command to the Arduino.

#         Args:
#             direction : 'left' | 'right' | None
#             intensity : 'none' | 'low' | 'high' | 'click'

#         Expected behaviour per intensity:
#             'none'  — send nothing
#             'low'   — short pulse  (~30ms)
#             'high'  — long pulse   (~80ms)
#             'click' — click event
#         """
#         pass

#     @abstractmethod
#     def reset(self) -> None:
#         """Called after each click attempt to reset the stimulation budget."""
#         pass

#     @abstractmethod
#     def close(self) -> None:
#         """Clean up serial connection on shutdown."""
#         pass