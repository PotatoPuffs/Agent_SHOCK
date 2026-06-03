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
from integration.interfacing import (BaseCNNObserver, BaseEMSController, SCREEN_W, TARGET_RADIUS, MAX_DX)
from integration.vision_hsv import DEFAULT_REGION

MEAN_PEAK    = 15.0   # px — average rightward displacement per pulse
STD_PEAK     = 6.0    # px — std of rightward displacements
MEAN_TROUGH  = 12.0   # px — average leftward displacement per pulse
STD_TROUGH   = 5.0    # px — std of leftward displacements
P_NO_RESP    = 0.05   # probability a pulse produces zero movement (fatigue)

class HSVBasedObserver(BaseCNNObserver):
    """
    Live HSV-based observer — real screen capture, no CNN.
 
    Stand-in for the RealCNNObserver while it is being finalised and to test real
    game live streaming/RL agent behaviours and outputs to EMS.
    Each get_state() grabs the game viewport with mss, runs the HSV
    detection in integration.vision to locate the red target and the crosshair,
    and returns the 4-float observation in the training contract's coordinate
    space.
 
    Coordinate normalisation
    ------------------------
    Detection happens in *capture-pixel* space (e.g. 1920 wide), but the model
    was trained in the contract's SCREEN_W space (1280) with TARGET_RADIUS and
    MAX_DX defined there. So detected positions are scaled by
    (screen_w / capture_width) before being returned. The normalised obs values
    are ratios and therefore resolution-independent regardless.

    """
 
    # Capture region defaults — game viewport only, excluding browser chrome. (easiest to put into full screen and use full screen resolution)
    # Mirrors collect_data.py (GAME_TOP / GAME_LEFT / GAME_WIDTH / GAME_HEIGHT).
 
    def __init__(
        self,
        screen_w: int            = SCREEN_W,
        capture_region: dict     = DEFAULT_REGION,
        detect_crosshair: bool   = True,
    ):
        """
        Args:
            screen_w         : contract width to scale detected positions into
                               (defaults to SCREEN_W = 1280, what the model trained on).
            capture_region   : mss region dict {top,left,width,height}. Defaults to
                               DEFAULT_REGION; override to match your monitor/browser.
            detect_crosshair : if True, locate the crosshair via green-pixel search;
                               if False, assume it sits at the viewport centre
                               (the crosshair barely moves in drill #52502).
        """
        import mss  # lazy — only needed for the live pathway
 
        self.screen_w         = screen_w
        self.capture_region   = dict(capture_region)
        self.detect_crosshair = detect_crosshair
 
        self._capture_w = float(self.capture_region["width"])
        self._scale     = self.screen_w / self._capture_w  # capture-px → contract-px
 
        self._sct = mss.mss()
 
        # State caches so detection drop-outs don't produce garbage observations
        self._last_target_x = self.screen_w / 2.0
        self._last_cursor_x = self.screen_w / 2.0
        self._have_target   = False
        self._cursor_x = self.screen_w / 2.0
        self._target_x = self.screen_w / 2.0
 
        print(f"[HSVObserver] Live HSV observer active — region={self.capture_region}, "
              f"scaling capture {int(self._capture_w)}px → contract {self.screen_w}px.")
 
    def _grab(self):
        """Capture the viewport and return (bgr, hsv) numpy frames."""
        import cv2  # lazy
        shot = self._sct.grab(self.capture_region)
        # mss returns BGRA bytes; build the array directly (no PIL needed)
        bgra = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(
            shot.height, shot.width, 4
        )
        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return bgr, hsv
 
    def reset_target(self) -> None:
        """No-op for the live observer — the real game manages its own targets.
 
        Provided for API parity with SimulatedCNNObserver so callers that
        reset on a hit don't error.
        """
        pass
 
    def get_state(
        self,
        last_dx: float,
        pulse_duration_ms: float,
    ) -> tuple[np.ndarray, float, float]:
        from integration.vision_hsv import find_target, find_crosshair
 
        bgr, hsv = self._grab()
 
        # ── Target (red sphere) ───────────────────────────────────────────────
        target = find_target(hsv)
        if target is not None:
            tx_cap, _ = target
            target_x  = tx_cap * self._scale
            self._last_target_x = target_x
            self._have_target   = True
        else:
            # Detection drop-out: reuse last known position (centre if never seen)
            target_x = self._last_target_x
 
        # ── Cursor (crosshair) ────────────────────────────────────────────────
        if self.detect_crosshair:
            cx_cap, _ = find_crosshair(bgr)
            cursor_x  = cx_cap * self._scale
        else:
            cursor_x = self.screen_w / 2.0
        self._last_cursor_x = cursor_x
 
        # ── Build contract observation (ratios — resolution-independent) ──────
        norm_error   = (cursor_x - target_x) / self.screen_w
        norm_cursor  = cursor_x / self.screen_w
        last_dx_norm = float(np.clip(last_dx / MAX_DX, -1.0, 1.0))
        pulse_norm   = pulse_duration_ms / 1000.0
 
        obs = np.array(
            [norm_error, norm_cursor, last_dx_norm, pulse_norm],
            dtype=np.float32,
        )
        self._cursor_x = cursor_x
        self._target_x = target_x
        return obs, float(target_x), float(cursor_x)
 
    def debug_snapshot(self, path: str = "hsv_debug.png") -> str:
        """
        Grab one frame, annotate detected target/crosshair, and save it.
 
        Use this to verify your HSV ranges and capture region BEFORE wiring
        to the EMS hardware. Coordinates drawn are in capture-pixel space.
        """
        import cv2  # lazy
        from integration.vision_hsv import find_target, find_crosshair
 
        bgr, hsv = self._grab()
        target = find_target(hsv)
        cx_cap, cy_cap = find_crosshair(bgr)
 
        cv2.drawMarker(bgr, (cx_cap, cy_cap), (0, 255, 0),
                       cv2.MARKER_CROSS, 30, 2)
        if target is not None:
            tx_cap, ty_cap = target
            cv2.circle(bgr, (tx_cap, ty_cap), 12, (0, 0, 255), 2)
            cv2.line(bgr, (cx_cap, cy_cap), (tx_cap, ty_cap), (255, 255, 0), 1)
            label = f"target=({tx_cap},{ty_cap})  cross=({cx_cap},{cy_cap})"
        else:
            label = f"NO TARGET  cross=({cx_cap},{cy_cap})"
        cv2.putText(bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
 
        cv2.imwrite(path, bgr)
        print(f"[HSVObserver] debug snapshot saved → {path}  ({label})")
        return path
 
    def close(self) -> None:
        """Release the mss capture handle."""
        try:
            self._sct.close()
        except Exception:
            pass
        print("[HSVObserver] Closed capture handle.")

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

    # def send_action_hsv(self, direction: str | None, intensity: str) -> float:
    def send_action_hsv(self, action: str) -> float:
        """
        Simulate the EMS pulse and update the observer's cursor position.

        Returns:
            actual_dx : signed pixel displacement (positive=right, negative=left).
                        For 'click' and 'none', returns 0.0.
                        run.py stores this and passes it back as last_dx.
        """
        actual_dx = 0.0

        if action == "right":
            # dx = self._sample_dx("right")
            # new_x = self.observer._cursor_x + dx
            # self.observer.update_cursor(new_x)
            # actual_dx = dx
            # self._pulse_count += 1
            print(f"[SimEMS] → RIGHT ")
                #   f"cursor={self.observer._cursor_x:.0f}px  "
                #   f"(pulse #{self._pulse_count})")

        elif action == "left":
            # dx = self._sample_dx("left")
            # new_x = self.observer._cursor_x - dx
            # self.observer.update_cursor(new_x)
            # actual_dx = -dx
            # self._pulse_count += 1
            print(f"[SimEMS] → LEFT ")
                #   f"cursor={self.observer._cursor_x:.0f}px  "
                #   f"(pulse #{self._pulse_count})")

        elif action == "click":
            # self._pulse_count += 1
            print(f"[SimEMS] → CLICK (momentary pulse) (pulse #{self._pulse_count})")

        elif action == "none":
            print(f"[SimEMS] → NONE (all relays open)")

        return actual_dx

    def close(self) -> None:
        print(f"[SimEMS] Closed. Total pulses sent: {self._pulse_count}")

class HSVEMSController(BaseEMSController):
    """
    Fake EMS — simulates stochastic hand movement instead of firing relays.

    Each pulse samples displacement from a Gaussian fitted to real EMS data
    so that the simulated deployment loop behaves like the real one.
    Returns actual_dx so run.py can pass it back as last_dx next frame.
    """

    def __init__(self, observer: HSVBasedObserver, std_scale: float = 1.0):
        """
        Args:
            observer  : HSVBasedObserver — updated after each pulse so
                        the obs reflects the simulated hand position.
            std_scale : 0.0 = deterministic mean movement, 1.0 = full variance.
                        Curriculum scheduler in train can ramp this externally.
        """
        self.observer   = observer
        self.std_scale  = std_scale
        self._pulse_count = 0
        print("[SimEMS] Simulated EMS active — no hardware connected.")

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
            print(f"[SimEMS] → RIGHT ")

        elif action == "left":
            print(f"[SimEMS] → LEFT ")

        elif action == "click":
            print(f"[SimEMS] → CLICK (momentary pulse) (pulse #{self._pulse_count})")

        elif action == "none":
            print(f"[SimEMS] → NONE (all relays open)")

        return actual_dx

    def close(self) -> None:
        print(f"[SimEMS] Closed. Total pulses sent: {self._pulse_count}")
