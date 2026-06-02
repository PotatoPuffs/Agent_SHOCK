"""
perception/cnn_observer.py — Real CNN observer for live deployment.

Implements BaseCNNObserver. Called every frame by run.py's deploy_loop().

Pipeline each call to get_state():
    1. Screenshot the game region via mss (fast, no focus steal)
    2. Pass the frame to the CNN to detect target_x and cursor_x
    3. Build and return the 4-float obs vector

Your CNN teammate fills in _detect_positions() with their model.
Everything else (obs building, normalisation, screen capture) is done here
and must NOT be changed — it's the contract with the RL agent.

Dependencies:
    pip install mss opencv-python numpy

How to find your game capture region:
    Run `python -c "import pyautogui; print(pyautogui.position())"` 
    while hovering over the top-left and bottom-right corners of the
    aiming.pro game area. Set CAPTURE_REGION below to match.
"""

import numpy as np
import cv2
import mss
import mss.tools
from interfacing import (
    BaseCNNObserver,
    SCREEN_W, MAX_DX, OBS_SIZE,
)

# ── Capture region ────────────────────────────────────────────────────────────
# Set these to the pixel coordinates of the aiming.pro game window on screen.
# top / left: top-left corner of the game area
# width / height: size of the capture region
# These do NOT have to match SCREEN_W — the frame is resized to SCREEN_W internally.
#
# To find your values:
#   python -c "import time, pyautogui; time.sleep(2); print(pyautogui.position())"
#   hover over top-left corner of game → note (x, y)
#   hover over bottom-right corner of game → note (x2, y2)
#   width = x2 - x, height = y2 - y
#
CAPTURE_REGION = {
    "top":    130,     # ❗ REAL VALUE: y coordinate of game top-left corner
    "left":   0,       # ❗ REAL VALUE: x coordinate of game top-left corner
    "width":  1280,    # ❗ REAL VALUE: width of game capture area
    "height": 720,     # ❗ REAL VALUE: height of game capture area
}
# ─────────────────────────────────────────────────────────────────────────────


class RealCNNObserver(BaseCNNObserver):
    """
    Live CNN observer — captures the game screen and detects target + cursor.

    Your CNN teammate implements _detect_positions() below.
    Everything else is fixed infrastructure.
    """

    def __init__(self, screen_w: int = SCREEN_W):
        self.screen_w  = screen_w
        self._sct      = mss.mss()              # screen capture handle (reused each frame)
        self._region   = CAPTURE_REGION

        # ── Load your CNN model here ──────────────────────────────────────
        # Replace this block with however your teammate loads their model.
        # Examples:
        #   self._model = torch.load("models/detector.pt").eval()
        #   self._model = tf.saved_model.load("models/detector")
        #   self._model = cv2.dnn.readNetFromONNX("models/detector.onnx")
        #
        self._model = None  # ❗ REPLACE: load your trained CNN here
        # ─────────────────────────────────────────────────────────────────

        print(f"[RealCNN] Capturing region: {self._region}")
        if self._model is None:
            print("[RealCNN] WARNING: _model is None — _detect_positions() will raise.")

    # ── Screen capture ────────────────────────────────────────────────────────

    def _grab_frame(self) -> np.ndarray:
        """
        Grab one frame from the game region.
        Returns a BGR uint8 numpy array of shape (height, width, 3).
        Resized to (SCREEN_W, *) so positions are in SCREEN_W pixel space.
        """
        raw   = self._sct.grab(self._region)
        frame = np.array(raw)[:, :, :3]              # BGRA → BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize width to SCREEN_W so detected x coords match the obs contract
        if frame.shape[1] != self.screen_w:
            scale  = self.screen_w / frame.shape[1]
            h_new  = int(frame.shape[0] * scale)
            frame  = cv2.resize(frame, (self.screen_w, h_new))

        return frame

    # ── Detection — IMPLEMENT THIS ────────────────────────────────────────────

    def _detect_positions(self, frame: np.ndarray) -> tuple[float, float]:
        """
        Run the CNN on one frame and return (target_x, cursor_x) in pixels.

        Args:
            frame : RGB uint8 array shape (height, SCREEN_W, 3)

        Returns:
            target_x : x position of the target centre in pixels
            cursor_x : x position of the cursor / crosshair in pixels

        ❗ THIS IS WHERE YOUR CNN TEAMMATE'S CODE GOES.

        Example skeleton for a PyTorch model:
        ----------------------------------------
        import torch, torchvision.transforms as T

        transform = T.Compose([T.ToTensor(), T.Resize((224, 224))])
        tensor    = transform(frame).unsqueeze(0)          # (1, 3, 224, 224)

        with torch.no_grad():
            pred = self._model(tensor)                     # model output shape depends on arch

        # If your model outputs [target_x_norm, cursor_x_norm] in [0, 1]:
        target_x = float(pred[0, 0]) * self.screen_w
        cursor_x = float(pred[0, 1]) * self.screen_w
        return target_x, cursor_x

        Example skeleton for OpenCV template / colour detection (fallback):
        -----------------------------------------------------------------------
        # Blue target circle — find the brightest blue blob
        hsv       = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask_tgt  = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
        M         = cv2.moments(mask_tgt)
        target_x  = M["m10"] / M["m00"] if M["m00"] > 0 else self.screen_w / 2

        # Green crosshair — find brightest green vertical line
        mask_cur  = cv2.inRange(hsv, (40, 100, 100), (80, 255, 255))
        cols      = mask_cur.sum(axis=0)
        cursor_x  = float(np.argmax(cols))
        return target_x, cursor_x
        -----------------------------------------------------------------------
        """
        raise NotImplementedError(
            "_detect_positions() is not implemented yet.\n"
            "Add your CNN detection code here in perception/cnn_observer.py."
        )

    # ── Public interface (called by run.py every frame) ───────────────────────

    def get_state(
        self,
        last_dx: float,
        pulse_duration_ms: float,
    ) -> tuple[np.ndarray, float, float]:
        """
        Capture frame → detect positions → build obs vector.

        Args:
            last_dx           : signed px displacement from the previous pulse.
                                Tracked by run.py; passed in so we can build obs[2].
            pulse_duration_ms : ms duration of the last pulse (constant in practice).
                                Passed in so we can build obs[3].

        Returns:
            obs      : np.ndarray (4,) float32 — ready to feed into the RL model
            target_x : float px — for run.py to judge click success
            cursor_x : float px — for run.py to judge click success
        """
        frame              = self._grab_frame()
        target_x, cursor_x = self._detect_positions(frame)

        # Clamp detected positions to valid range
        target_x = float(np.clip(target_x, 0, self.screen_w))
        cursor_x = float(np.clip(cursor_x, 0, self.screen_w))

        # Build obs — identical formula to aiming_env._build_obs() and SimulatedCNNObserver
        norm_error    = (cursor_x - target_x) / self.screen_w
        norm_cursor   = cursor_x / self.screen_w
        last_dx_norm  = float(np.clip(last_dx / MAX_DX, -1.0, 1.0))
        pulse_norm    = pulse_duration_ms / 1000.0

        obs = np.array(
            [norm_error, norm_cursor, last_dx_norm, pulse_norm],
            dtype=np.float32,
        )

        assert obs.shape == (OBS_SIZE,), f"[RealCNN] obs shape error: {obs.shape}"
        return obs, target_x, cursor_x

    def close(self) -> None:
        self._sct.close()
        print("[RealCNN] Screen capture closed.")