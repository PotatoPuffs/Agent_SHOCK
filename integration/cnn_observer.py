"""
perception/cnn_observer.py — Real CNN observer for live deployment.

Implements BaseCNNObserver. Called every frame by run.py's deploy_loop().

Integrates the AgentShockCNN model from cnn/inference.py:
    1. Screenshot the game region via mss (fast, no focus steal)
    2. Preprocess frame (resize, normalize) to match CNN training
    3. Run CNN forward pass → detect [cursor_x, cursor_y, target_x, target_y]
    4. Denormalize to pixel coordinates
    5. Return (target_x, cursor_x) to build obs vector

Pipeline each call to get_state():
    1. Screenshot the game region via mss (fast, no focus steal)
    2. Pass the frame to the CNN to detect target_x and cursor_x
    3. Build and return the 4-float obs vector

Dependencies:
    pip install mss torch pillow torchvision numpy

How to find your game capture region:
    For browser-based game (e.g., aiming.pro in Chrome):
    1. Open the game in a browser at full resolution
    2. Run: python -c "import pyautogui; print(pyautogui.position())"
    3. Hover over top-left corner of game area → note (x, y)
    4. Hover over bottom-right corner → note (x2, y2)
    5. Set: width = x2 - x, height = y2 - y, top = y, left = x
"""

import numpy as np
import torch
import os
import sys
from PIL import Image
import torchvision.transforms as T
import mss
from integration.interfacing import (
    BaseCNNObserver,
    SCREEN_W, MAX_DX, OBS_SIZE,
)

# Import the CNN model architecture
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cnn'))
from cnn.model import AgentShockCNN

# ─────────────────────────────────────────────────────────────────────────────
# CNN INFERENCE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# These must match the CNN training configuration in cnn/train.py

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'cnn', 'checkpoints', 'best_cnn.pth')

# CNN input dimensions — MUST match training config (cnn/train.py CONFIG)
CNN_INPUT_H = 224
CNN_INPUT_W = 224

# Frame resolution used for denormalisation
# ❗ SET TO MATCH YOUR CAPTURE REGION RESOLUTION
FRAME_W = 1280
FRAME_H = 720

# Select GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing transform — MUST be IDENTICAL to training (cnn/dataset.py EVAL_TRANSFORMS)
# Mismatch here is a common silent bug: model will receive different input distribution
INFERENCE_TRANSFORM = T.Compose([
    T.Resize((CNN_INPUT_H, CNN_INPUT_W)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

# ── Capture region ────────────────────────────────────────────────────────────
# Browser game window coordinates (for aiming.pro or similar game running in browser)
# Set these to the pixel coordinates of the game window on screen.
# top / left: top-left corner of the game area
# width / height: size of the capture region
#
# To find your values:
#   1. Open the game in browser (e.g., aiming.pro in Chrome)
#   2. Run: python -c "import time, pyautogui; time.sleep(2); print(pyautogui.position())"
#   3. Hover over top-left corner of game area → note position
#   4. Hover over bottom-right corner → note position
#   5. Calculate: width = x2 - x1, height = y2 - y1
#
CAPTURE_REGION = {
    "top":    0,     # ❗ REAL VALUE: y coordinate of game top-left corner
    "left":   0,       # ❗ REAL VALUE: x coordinate of game top-left corner
    "width":  FRAME_W, # ❗ REAL VALUE: width of game capture area (e.g., 1280)
    "height": FRAME_H, # ❗ REAL VALUE: height of game capture area (e.g., 720)
}
# ─────────────────────────────────────────────────────────────────────────────


class RealCNNObserver(BaseCNNObserver):
    """
    Live CNN observer — captures game screen and detects cursor + target via AgentShockCNN.

    Pipeline each frame:
        1. Capture screenshot from game window (mss)
        2. Preprocess: resize & normalize to match CNN training
        3. CNN forward pass → [Cx_norm, Cy_norm, Tx_norm, Ty_norm]
        4. Denormalize to pixel coordinates
        5. Return (target_x, cursor_x) for obs building
    """

    def __init__(self, screen_w: int = SCREEN_W):
        self.screen_w  = screen_w
        self._sct      = mss.mss()              # screen capture handle (reused each frame)
        self._region   = CAPTURE_REGION

        # ── Load the trained CNN model ────────────────────────────────────────
        print(f"[RealCNN] Loading CNN model from {MODEL_PATH}...")
        self._model = self._load_model(MODEL_PATH)
        print(f"[RealCNN] CNN model loaded successfully on device: {DEVICE}")

        print(f"[RealCNN] Capturing region: {self._region}")
        print(f"[RealCNN] Frame resolution (for denormalization): {FRAME_W}×{FRAME_H}px")

    # ── CNN Model Loading ─────────────────────────────────────────────────────

    def _load_model(self, model_path: str) -> AgentShockCNN:
        """
        Load trained AgentShockCNN from checkpoint file.

        Args:
            model_path : path to .pth checkpoint file

        Returns:
            model : AgentShockCNN in eval mode, weights loaded, on DEVICE
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[RealCNN] Model checkpoint not found: {model_path}\n"
                f"  Run cnn/train.py first to train and save the model."
            )

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE)

        # Rebuild model architecture
        model = AgentShockCNN(CNN_INPUT_H, CNN_INPUT_W).to(DEVICE)

        # Copy trained weights
        model.load_state_dict(checkpoint["model_state"])

        # Switch to inference mode (disable dropout, batchnorm stochasticity)
        model.eval()

        print(f"  ✓ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} | "
              f"val_loss={checkpoint.get('val_loss', '?'):.6f}")

        return model

    # ── Screen Capture ────────────────────────────────────────────────────────

    def _capture_frame(self) -> Image.Image:
        """
        Capture a screenshot of the game region and return as PIL Image (RGB).

        Returns:
            PIL.Image.Image in RGB mode, ready for preprocessing
        """
        screenshot = self._sct.grab(self._region)   # raw BGRA buffer
        img = Image.frombytes(
            "RGB",
            screenshot.size,
            screenshot.bgra,
            "raw",
            "BGRX"  # decode BGRA, ignore alpha → RGB
        )
        # img.show()
        img.save("debug_capture.png")
        return img

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _preprocess_frame(self, frame: Image.Image) -> torch.Tensor:
        """
        Apply INFERENCE_TRANSFORM to frame and add batch dimension.

        Args:
            frame : PIL.Image.Image in RGB mode

        Returns:
            torch.Tensor shape (1, 3, 224, 224) — ready for CNN
        """
        tensor = INFERENCE_TRANSFORM(frame)       # (3, 224, 224)
        return tensor.unsqueeze(0)                 # (1, 3, 224, 224) — add batch dim

    # ── CNN Inference ─────────────────────────────────────────────────────────

    def _predict_coordinates(self, frame_tensor: torch.Tensor) -> tuple[float, float]:
        """
        Run CNN forward pass and return (target_x, cursor_x) in pixel space.

        Args:
            frame_tensor : torch.Tensor shape (1, 3, 224, 224)

        Returns:
            target_x : x coordinate of target centre in pixels
            cursor_x : x coordinate of cursor/crosshair in pixels
        """
        with torch.no_grad():
            # Move tensor to model's device
            output = self._model(frame_tensor.to(DEVICE))

            # Remove batch dimension and convert to numpy
            output = output.squeeze(0).cpu().numpy()

        # Denormalise: multiply by frame resolution
        # CNN output order: [Cx_norm, Cy_norm, Tx_norm, Ty_norm]
        cx_px = float(output[0]) * FRAME_W   # cursor x in pixels
        cy_px = float(output[1]) * FRAME_H   # cursor y in pixels
        tx_px = float(output[2]) * FRAME_W   # target x in pixels
        ty_px = float(output[3]) * FRAME_H   # target y in pixels

        # For the obs contract we only return x coordinates
        # (y is implicit in the capture region)
        return tx_px, cx_px

    # ── Detection (called by get_state) ───────────────────────────────────────

    def _detect_positions(self, frame: Image.Image) -> tuple[float, float]:
        """
        Run the CNN on one frame and return (target_x, cursor_x) in pixels.

        Args:
            frame : PIL.Image.Image in RGB mode

        Returns:
            target_x : x position of the target centre in pixels
            cursor_x : x position of the cursor / crosshair in pixels
        """
        # Preprocess and run CNN
        tensor = self._preprocess_frame(frame)
        target_x, cursor_x = self._predict_coordinates(tensor)

        return target_x, cursor_x

    # ── Public interface (called by run.py every frame) ───────────────────────

    def get_state(
        self,
        last_dx: float,
        pulse_duration_ms: float,
    ) -> tuple[np.ndarray, float, float]:
        """
        Capture frame → CNN → detect positions → build obs vector.

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
        # Capture and run CNN
        frame              = self._capture_frame()
        target_x, cursor_x = self._detect_positions(frame)

        # Scale detected positions from FRAME_W → SCREEN_W coordinate space
        # CNN detects in full frame (FRAME_W resolution), but obs is in SCREEN_W
        scale_x = self.screen_w / FRAME_W
        target_x_scaled = target_x * scale_x
        cursor_x_scaled = cursor_x * scale_x

        # Clamp to valid range
        target_x_scaled = float(np.clip(target_x_scaled, 0, self.screen_w))
        cursor_x_scaled = float(np.clip(cursor_x_scaled, 0, self.screen_w))

        # Build obs — identical formula to aiming_env._build_obs() and SimulatedCNNObserver
        norm_error    = (cursor_x_scaled - target_x_scaled) / self.screen_w
        norm_cursor   = cursor_x_scaled / self.screen_w
        last_dx_norm  = float(np.clip(last_dx / MAX_DX, -1.0, 1.0))
        pulse_norm    = pulse_duration_ms / 1000.0

        obs = np.array(
            [norm_error, norm_cursor, last_dx_norm, pulse_norm],
            dtype=np.float32,
        )

        assert obs.shape == (OBS_SIZE,), f"[RealCNN] obs shape error: {obs.shape}"
        return obs, target_x_scaled, cursor_x_scaled

    def close(self) -> None:
        self._sct.close()
        print("[RealCNN] Screen capture closed.")