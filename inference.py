"""
inference.py — Live Inference Phase for Agent Shock CNN
=======================================================
WHAT THIS FILE DOES:
  Implements the LIVE inference loop shown in the project slides:

      Capture Frame → Preprocess → CNN Forward Pass → Compute Δx/Δy → RL Agent

  This runs continuously during gameplay on Aiming.Pro.
  Each frame the CNN outputs [Cx, Cy, Tx, Ty], from which we compute:
      Δx = Tx - Cx   (signed horizontal error → positive means target is RIGHT)
      Δy = Ty - Cy   (signed vertical error   → positive means target is BELOW)
  These values are passed to the RL agent which decides the EMS pulse action.

TUTORIAL LINK (Tutorial 08):
  The Tutorial 08 pattern for inference was:
      model = fasterrcnn_resnet50_fpn(weights=weights)
      model = model.eval()           # switch to inference mode
      with torch.no_grad():          # disable gradient tracking
          outputs = model(inputs)    # forward pass

  We follow exactly the same pattern, just with our custom AgentShockCNN
  and a real-time screen capture loop instead of a static image list.

KEY LIBRARIES:
  mss            — ultra-fast cross-platform screen capture (pip install mss)
                   directly accesses OS display buffers → ~30-60 FPS
  PIL (Pillow)   — image manipulation and format conversion
  torch          — CNN inference
  torchvision.transforms — identical preprocessing pipeline to training
"""

import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

# mss: "Multiple ScreenShots" — fastest Python screen capture library.
# Unlike pyautogui or PIL.ImageGrab, mss directly reads OS screen buffers
# in BGRA format without going through GUI APIs → ~30-60 FPS on most systems.
import mss

from model import AgentShockCNN


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — must match collect_data.py and train.py CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "checkpoints/best_cnn.pth"  # saved by train.py

FRAME_W    = 1920   # actual game/screen resolution (used to de-normalise coords)
FRAME_H    = 1080

INPUT_H    = 224    # CNN input size — MUST match CONFIG in train.py
INPUT_W    = 224    # changing this requires re-training

# Screen capture region — set to the game window bounds.
# Full screen example:  {"top": 0, "left": 0, "width": 1920, "height": 1080}
# Cropped to game only: {"top": 110, "left": 0, "width": 1920, "height": 970}
CAPTURE_REGION = {"top": 0, "left": 0, "width": FRAME_W, "height": FRAME_H}

# Select GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────
#
# CRITICAL: This must be IDENTICAL to EVAL_TRANSFORMS in dataset.py.
# If the preprocessing at inference time differs from training time, the CNN
# receives inputs with a different distribution than it was trained on →
# coordinates will be wrong (a common silent bug).
#
# Same steps as EVAL_TRANSFORMS:
#   1. Resize to 224×224 (fixed CNN input size)
#   2. ToTensor (PIL HWC uint8 → CHW float32 [0,1])
#   3. Normalize with ImageNet mean/std (matches training distribution)

INFERENCE_TRANSFORM = T.Compose([
    T.Resize((INPUT_H, INPUT_W)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load Model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str) -> AgentShockCNN:
    """
    Loads a trained AgentShockCNN from a checkpoint file (.pth).

    TUTORIAL LINK:
      In Tutorial 08 you loaded pre-trained weights with:
          model = fasterrcnn_resnet50_fpn(weights=weights)
      Here we load OUR trained weights using torch.load + load_state_dict:
          checkpoint = torch.load(path)           # deserialise the dict
          model.load_state_dict(checkpoint["model_state"])  # copy weights in

    torch.load()
      Deserialises the checkpoint dict that train.py saved with torch.save().
      map_location=DEVICE ensures weights load to the correct device even
      if the model was trained on GPU but we're now running on CPU (or vice versa).

    model.load_state_dict()
      Copies the saved weight tensors into the model's matching parameter slots.
      The keys must match — this is why we save model.state_dict() (not the
      whole model object) and reconstruct AgentShockCNN with the same architecture.

    model.eval()
      MUST be called before inference. Switches off:
        • Dropout (would randomly zero neurons → non-deterministic output)
        • BatchNorm training mode (would use batch statistics → unstable output)
      Without this, running the same frame twice gives different coordinates.
      In Tutorial 08 every model was set to .eval() before inference.

    Args:
        model_path : path to checkpoint file (e.g. "checkpoints/best_cnn.pth")

    Returns:
        model : AgentShockCNN in eval mode, weights loaded, on DEVICE
    """
    # Load the full checkpoint dictionary saved by train.py
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Rebuild the model architecture (empty, randomly initialised weights)
    model = AgentShockCNN(INPUT_H, INPUT_W).to(DEVICE)

    # Copy the trained weights into the model
    # strict=True (default) — every key in state_dict must match the model
    model.load_state_dict(checkpoint["model_state"])

    # Switch to inference mode — CRITICAL (see docstring above)
    model.eval()

    print(f"[load_model] Loaded checkpoint from epoch {checkpoint['epoch']} | "
          f"val_loss={checkpoint['val_loss']:.6f} | "
          f"pixel_error={checkpoint['px_err']:.2f}px")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Capture Screen Frame
# ─────────────────────────────────────────────────────────────────────────────

def capture_frame(sct: mss.mss, region: dict) -> Image.Image:
    """
    Captures a screenshot of the specified screen region using mss.

    mss.grab(region)
      Returns a ScreenShot object containing raw pixel data in BGRA format
      (Blue-Green-Red-Alpha — note the reversed channel order vs RGB).
      Reading directly from the OS frame buffer makes this ~5× faster than
      PIL.ImageGrab.grab() which goes through GDI/X11 APIs.

    Image.frombytes()
      Constructs a PIL Image from the raw byte buffer.
      We decode "BGRX" (Blue-Green-Red-ignore_Alpha) and output "RGB".

    Why reuse the same sct context?
      Opening a new mss context (mss.mss()) each call re-initialises the
      OS screen reader. Reusing one context across the loop avoids that
      overhead and maintains ~60 FPS capture.

    Args:
        sct    : mss.mss() context (created once in run_inference_loop)
        region : {"top": Y, "left": X, "width": W, "height": H}

    Returns:
        PIL.Image.Image in RGB mode
    """
    screenshot = sct.grab(region)   # raw BGRA pixel buffer from OS
    img = Image.frombytes(
        "RGB",
        screenshot.size,       # (width, height) tuple
        screenshot.bgra,       # raw bytes
        "raw", "BGRX"          # decoder: Blue-Green-Red-ignore_Alpha → RGB
    )
    return img


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Preprocess Frame
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_frame(frame: Image.Image) -> torch.Tensor:
    """
    Applies INFERENCE_TRANSFORM to a single PIL Image, adds batch dimension.

    TUTORIAL LINK:
      In Tutorial 08 transforms were applied per image:
          inputs = [transforms(d) for d in image_list]
          batch_input = torch.stack(inputs)      # list → (N, 3, H, W) tensor
      Here we process ONE frame at a time and use unsqueeze(0) to add the
      batch dimension of 1, since we're running real-time frame-by-frame.

    After INFERENCE_TRANSFORM:
      PIL Image (1920×1080, RGB, uint8)
        → resize → (224×224, RGB, uint8)
        → ToTensor → (3, 224, 224, float32, [0,1])
        → Normalize → (3, 224, 224, float32, ImageNet-normalised)

    unsqueeze(0):
      The CNN's forward() expects input shape (batch_size, 3, H, W).
      A single image is (3, H, W) — we add the missing batch dimension:
      (3, 224, 224) → (1, 3, 224, 224)

    Args:
        frame : PIL.Image.Image — raw RGB screenshot

    Returns:
        torch.Tensor shape (1, 3, 224, 224) — ready for CNN forward pass
    """
    tensor = INFERENCE_TRANSFORM(frame)   # (3, 224, 224)
    return tensor.unsqueeze(0)            # (1, 3, 224, 224)  — add batch dim


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 & 5: CNN Forward Pass + Compute Coordinate Difference
# ─────────────────────────────────────────────────────────────────────────────

def predict_coordinates(model: AgentShockCNN,
                         frame_tensor: torch.Tensor) -> dict:
    """
    Runs one CNN forward pass and converts normalised outputs → pixel errors.

    TUTORIAL LINK:
      Tutorial 08 inference pattern — replicated here:
          model.eval()                          # done once in load_model()
          with torch.no_grad():                 # disable gradient tracking
              outputs = model(inputs)           # forward pass

    torch.no_grad()
      Context manager: temporarily disables autograd's computation graph.
      During inference we never call .backward(), so tracking gradients
      wastes memory and time. This block reduces memory by ~30% and
      speeds up forward passes noticeably at high frame rates.

    Denormalisation:
      CNN output is in [0, 1]. To get pixel coordinates:
          cx_px = cx_norm × FRAME_W
          cy_px = cy_norm × FRAME_H
      Then the coordinate difference:
          Δx = tx_px - cx_px    (signed: positive → aim RIGHT)
          Δy = ty_px - cy_px    (signed: positive → aim DOWN)

    Args:
        model        : AgentShockCNN (eval mode, loaded by load_model)
        frame_tensor : (1, 3, 224, 224) preprocessed frame on CPU or GPU

    Returns:
        dict with keys:
          cx, cy    — crosshair pixel position
          tx, ty    — target    pixel position
          delta_x   — signed horizontal error (Tx - Cx) in pixels
          delta_y   — signed vertical   error (Ty - Cy) in pixels
    """
    with torch.no_grad():
        # Move tensor to model's device before forward pass
        output = model(frame_tensor.to(DEVICE))   # → (1, 4) normalised

        # squeeze(0): removes the batch dimension → (4,) 1D array
        # .cpu(): moves back to CPU before converting to numpy
        # .numpy(): converts to numpy array for arithmetic
        output = output.squeeze(0).cpu().numpy()   # → (4,) numpy array

    # Denormalise: multiply by screen resolution to get pixel coordinates
    # output order: [Cx_norm, Cy_norm, Tx_norm, Ty_norm]
    cx = float(output[0]) * FRAME_W    # crosshair x in pixels
    cy = float(output[1]) * FRAME_H    # crosshair y in pixels
    tx = float(output[2]) * FRAME_W    # target    x in pixels
    ty = float(output[3]) * FRAME_H    # target    y in pixels

    # ── Coordinate Difference — core output sent to the RL agent ──────
    # Δx > 0 : target is to the RIGHT of crosshair → stimulate rightward
    # Δx < 0 : target is to the LEFT  of crosshair → stimulate leftward
    # Δy > 0 : target is BELOW  the crosshair → (used for vertical EMS if needed)
    # Δy < 0 : target is ABOVE  the crosshair
    delta_x = tx - cx
    delta_y = ty - cy

    #print(f"[INF] Crosshair: ({cx:.2f}, {cy:.2f}) | Target: ({tx:.2f}, {ty:.2f}) | Delta: (Δx: {delta_x:+.2f}, Δy: {delta_y:+.2f})")

    return {
        "cx"      : cx,
        "cy"      : cy,
        "tx"      : tx,
        "ty"      : ty,
        "delta_x" : delta_x,
        "delta_y" : delta_y,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP: Live Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_loop(callback=None, target_fps: int = 30):
    """
    Main real-time inference loop — runs continuously during gameplay.

    Each iteration (one game frame):
      1. Capture current game frame from screen  [capture_frame]
      2. Preprocess frame                         [preprocess_frame]
      3. CNN forward pass → [Cx, Cy, Tx, Ty]     [predict_coordinates]
      4. Compute Δx, Δy                           [predict_coordinates]
      5. Call callback with error → RL agent      [rl_agent_callback]

    Frame rate control:
      target_fps=30 → frame_interval = 1/30 ≈ 33ms per frame.
      After each frame we sleep for the remaining time so we maintain
      approximately the target rate without spinning the CPU at 100%.

    Args:
        callback   : function(delta_x, delta_y, coords) called each frame.
                     Connect your RL agent / EMS controller here.
        target_fps : desired inference rate in frames per second.
    """
    model         = load_model(MODEL_PATH)
    frame_interval = 1.0 / target_fps

    # ── State memory — tracks previous frame values ───────────────
    prev_coords = None   # coords dict from last frame
    prev_action = 0      # last EMS action taken

    with mss.MSS() as sct:
        frame_count = 0

        while True:
            loop_start = time.perf_counter()

            # ── Capture + preprocess + CNN forward pass ────────────
            frame  = capture_frame(sct, CAPTURE_REGION)
            tensor = preprocess_frame(frame)
            coords = predict_coordinates(model, tensor)

            # ── Compute current distances ──────────────────────────
            delta_x    = coords["delta_x"]           # signed horizontal
            delta_y    = coords["delta_y"]           # signed vertical
            distance   = np.sqrt(delta_x**2 +
                                 delta_y**2)         # Euclidean distance

            # ── Compute previous frame movement ────────────────────
            if prev_coords is not None:
                # How much did the TARGET move since last frame?
                target_move_x = coords["tx"] - prev_coords["tx"]
                target_move_y = coords["ty"] - prev_coords["ty"]

                # How much did the CROSSHAIR move since last frame?
                cursor_move_x = coords["cx"] - prev_coords["cx"]
                cursor_move_y = coords["cy"] - prev_coords["cy"]

                # Direction of previous movement as a simple flag:
                # +1 = moved right, -1 = moved left, 0 = no movement
                prev_target_direction = np.sign(target_move_x)
                prev_cursor_direction = np.sign(cursor_move_x)
            else:
                # First frame — no previous data available yet
                target_move_x         = 0.0
                target_move_y         = 0.0
                cursor_move_x         = 0.0
                cursor_move_y         = 0.0
                prev_target_direction = 0
                prev_cursor_direction = 0

            # ── Build the full game state dict ─────────────────────
            # This is what gets passed to the RL agent every frame
            game_state = {
                # Current positions (pixels)
                "cx"                  : coords["cx"],
                "cy"                  : coords["cy"],
                "tx"                  : coords["tx"],
                "ty"                  : coords["ty"],

                # Signed component distances
                "delta_x"             : delta_x,
                "delta_y"             : delta_y,

                # Straight-line Euclidean distance
                "distance"            : distance,

                # How much each object moved since last frame
                "target_move_x"       : target_move_x,
                "target_move_y"       : target_move_y,
                "cursor_move_x"       : cursor_move_x,
                "cursor_move_y"       : cursor_move_y,

                # Direction flags: +1 right, -1 left, 0 stationary
                "prev_target_direction": prev_target_direction,
                "prev_cursor_direction": prev_cursor_direction,

                # Last action the RL agent took
                "prev_action"         : prev_action,
            }

            # ── Send full state to RL agent ────────────────────────
            if callback is not None:
                prev_action = callback(game_state)

            print(
                f"--- FRAME {frame_count} ---\n"
                f"Positions | Crosshair: ({game_state['cx']:.2f}, {game_state['cy']:.2f}) | Target: ({game_state['tx']:.2f}, {game_state['ty']:.2f})\n"
                f"Error     | Δx: {game_state['delta_x']:+.2f} | Δy: {game_state['delta_y']:+.2f} | Distance: {game_state['distance']:.2f}\n"
                f"Deltas    | Target Move: ({game_state['target_move_x']:+.2f}, {game_state['target_move_y']:+.2f}) | Cursor Move: ({game_state['cursor_move_x']:+.2f}, {game_state['cursor_move_y']:+.2f})\n"
                f"States    | Target Dir: {game_state['prev_target_direction']} | Cursor Dir: {game_state['prev_cursor_direction']} | Prev Action: {game_state['prev_action']}\n"
            )

            # ── Store this frame for next iteration ────────────────
            prev_coords = coords
            frame_count += 1

            # ── Frame rate limiter ─────────────────────────────────
            elapsed    = time.perf_counter() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


# ─────────────────────────────────────────────────────────────────────────────
# RL AGENT CALLBACK (stub — replace with your real DQN / PPO policy)
# ─────────────────────────────────────────────────────────────────────────────

def rl_agent_callback(game_state: dict) -> int:
    """
    Receives the full game state every frame.
    Returns the action taken (stored as prev_action for next frame).

    The RL state vector the agent sees:
      [delta_x, delta_y, distance,
       target_move_x, target_move_y,
       prev_target_direction, prev_action]
    """

    # Build the state vector your DQN/PPO policy expects
    state = np.array([
        game_state["delta_x"],
        game_state["delta_y"],
        game_state["distance"],
        game_state["target_move_x"],    # target velocity x
        game_state["target_move_y"],    # target velocity y
        game_state["prev_target_direction"],
        game_state["prev_action"],
    ])

    # Placeholder threshold policy — replace with trained RL policy:
    # action = dqn_policy.select_action(state)
    abs_error = abs(game_state["delta_x"])

    if abs_error < 5:
        action = 0
    elif abs_error < 30:
        action = 1
    elif abs_error < 80:
        action = 2
    else:
        action = 3

    return action   # returned so it becomes prev_action next frame


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start the live inference loop with the example RL agent stub.
    # Replace rl_agent_callback with your actual trained policy.
    run_inference_loop(callback=rl_agent_callback, target_fps=30)