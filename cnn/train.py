"""
train.py — Offline Training Phase for Agent Shock CNN
=====================================================
WHAT THIS FILE DOES:
  Implements the complete OFFLINE training loop shown in the project slides:

      Collect Frames → Label Coordinates → Train CNN → Validate → Save model

  After running this file you get  checkpoints/best_cnn.pth  — the serialised
  model weights that inference.py loads during live gameplay.

TUTORIAL LINK (Tutorial 08):
  In Tutorial 08 you loaded a PRE-TRAINED model:
      model = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
      model.eval()
      outputs = model(inputs)
  
  Here we TRAIN from scratch instead of using pre-trained weights:
      model = AgentShockCNN().to(DEVICE)
      for epoch in range(epochs):
          train_one_epoch(model, loader, criterion, optimizer)
  
  The key additions are:
    • criterion (loss function)  — measures how wrong the predictions are
    • optimizer                  — adjusts weights to reduce the loss
    • model.train() / model.eval() — switches training vs inference modes

KEY LIBRARIES:
  torch.nn     — loss functions (nn.MSELoss for regression)
  torch.optim  — optimisation algorithms (Adam, SGD, learning rate schedulers)
  torch.cuda   — GPU acceleration (automatic via DEVICE selection)
  tqdm         — progress bar showing per-batch training progress
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm   # pip install tqdm — wraps any iterable with a live progress bar

from model   import AgentShockCNN
from dataset import build_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Centralised dict — change values here; they propagate automatically.
# The entire CONFIG dict is also saved into the checkpoint so you always know
# which settings produced a given model file.

CONFIG = {
    "csv_path"       : "data/labels.csv",    # labels produced by collect_data.py
    "frames_dir"     : "data/frames/",        # frame PNGs from collect_data.py
    "checkpoint_dir" : "checkpoints/",        # where to save best_cnn.pth
    "batch_size"     : 32,    # frames per gradient update step
                               # ↑ higher = faster but needs more GPU memory
    "epochs"         : 50,    # full passes through the entire training set
                               # early stopping usually kicks in well before this
    "lr"             : 1e-3,  # learning rate: how large each weight update step is
                               # too high → loss oscillates; too low → trains slowly
    "weight_decay"   : 1e-4,  # L2 regularisation coefficient
                               # penalises large weights → helps generalisation
    "patience"       : 8,     # early stopping patience (epochs without improvement)
    "input_h"        : 224,   # CNN input height — must match dataset.py TRANSFORMS
    "input_w"        : 224,   # CNN input width
}


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
# PyTorch can run on CPU or GPU. NVIDIA GPU (CUDA) is typically 10–50× faster.
# torch.cuda.is_available() returns True if CUDA drivers + GPU are present.
# All tensors and the model must be on the SAME device — moving between devices
# mid-computation causes an error.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train.py] Training device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Pixel Error Metric
# ─────────────────────────────────────────────────────────────────────────────

def compute_pixel_error(pred: torch.Tensor,
                         target: torch.Tensor,
                         frame_w: int = 1920,
                         frame_h: int = 1080) -> float:
    """
    Converts normalised coordinate prediction error into interpretable PIXELS.

    Why do we need this?
      The CNN outputs values in [0, 1]. The training loss (MSE) is in that
      same normalised space — e.g. a loss of 0.0001 is hard to interpret.
      Pixel error translates this to: "on average, the CNN is off by X pixels"
      which is directly meaningful (a 5px error vs a 50px error).

    Computation:
      |Δx| in pixels = |pred_cx_norm - true_cx_norm| × frame_width
      |Δy| in pixels = |pred_cy_norm - true_cy_norm| × frame_height
      We compute this for all 4 coordinates and average over the whole batch.

    Args:
        pred   : (B, 4) — CNN predictions in normalised space
        target : (B, 4) — ground truth labels in normalised space
        frame_w, frame_h : screen resolution used to rescale back to pixels

    Returns:
        mean absolute pixel error (float) across batch and all 4 coordinates
    """
    # scale tensor: [frame_w, frame_h, frame_w, frame_h]
    # x-columns (0, 2) get multiplied by frame_w
    # y-columns (1, 3) get multiplied by frame_h
    scale = torch.tensor(
        [frame_w, frame_h, frame_w, frame_h],
        device=pred.device, dtype=torch.float32
    )

    # Element-wise absolute error, then multiply by pixel scale
    pixel_err = (pred - target).abs() * scale   # (B, 4) — errors in pixels

    # .mean() averages over both the batch dimension B and coordinate dimension 4
    # .item() extracts the single Python float from the scalar tensor
    return pixel_err.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING: One Epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer) -> float:
    """
    Runs one complete pass through the TRAINING data.

    One "epoch" = the model has seen every training frame exactly once
    (in random order, because DataLoader shuffle=True).

    Each batch iteration:
      1. Load batch (frames + coords) from DataLoader
      2. Zero gradients (clear last batch's accumulated gradients)
      3. Forward pass: CNN predicts coordinates
      4. Compute loss: how far off are predictions?
      5. Backward pass: compute gradients (which weights caused the error?)
      6. Clip gradients (prevent instability)
      7. Optimizer step: adjust weights to reduce the loss

    Args:
        model     : AgentShockCNN instance (in training mode)
        loader    : DataLoader yielding (frames, coords) batches
        criterion : loss function — nn.MSELoss()
        optimizer : weight update rule — optim.Adam(...)

    Returns:
        avg_loss (float) — mean training loss across all batches this epoch
    """

    # model.train() switches the model into TRAINING MODE:
    #   • Dropout layers are ACTIVE (randomly zero neurons)
    #   • BatchNorm uses batch statistics (not running averages)
    # In Tutorial 08 you saw model.eval() for inference;
    # model.train() is the inverse — always call before training loop.
    model.train()

    total_loss = 0.0

    # tqdm wraps the DataLoader iterator and shows a live progress bar:
    #   Train  |████████████████| 32/32 [00:05<00:00,  5.8it/s]
    for frames, coords in tqdm(loader, desc="  Train", leave=False):

        # ── Move data to GPU (or keep on CPU) ─────────────────────────
        # Tensors must be on the same device as the model.
        # .to(DEVICE) is a no-op if the tensor is already on DEVICE.
        frames = frames.to(DEVICE)    # (B, 3, 224, 224) — input frames
        coords = coords.to(DEVICE)    # (B, 4)           — ground-truth [Cx,Cy,Tx,Ty]

        # ── Step 1: Zero gradients ─────────────────────────────────────
        # PyTorch ACCUMULATES gradients by default across backward() calls.
        # We must reset them each batch so this batch's gradient doesn't
        # add to last batch's residual gradient.
        optimizer.zero_grad()

        # ── Step 2: Forward pass ───────────────────────────────────────
        # model(frames) calls AgentShockCNN.forward(frames) automatically.
        # preds shape: (B, 4) — each row is [Cx_norm, Cy_norm, Tx_norm, Ty_norm]
        preds = model(frames)

        # ── Step 3: Compute loss ───────────────────────────────────────
        # MSELoss (Mean Squared Error):
        #   loss = mean( (pred_i - target_i)² )  over all B×4 values
        #
        # Why MSE not MAE (Mean Absolute Error)?
        #   MSE squares the error, so large misses are penalised much more
        #   than small misses. This pushes the model hard to avoid big errors
        #   (which would cause the TENS to deliver a wrong-direction pulse).
        #
        # In Tutorial 08 you saw CrossEntropyLoss for segmentation (classes).
        # For REGRESSION (continuous coordinates), MSELoss is standard.
        loss = criterion(preds, coords)

        # ── Step 4: Backward pass ──────────────────────────────────────
        # loss.backward() uses PyTorch's autograd engine to compute:
        #   ∂loss/∂w  for every learnable weight w in the model
        # This is done via the chain rule through the computation graph
        # that was built during the forward pass.
        loss.backward()

        # ── Step 5: Gradient clipping ──────────────────────────────────
        # If gradients become very large ("exploding gradients"), weight
        # updates are unstable. clip_grad_norm_ rescales all gradients so
        # their combined L2 norm is at most max_norm=1.0.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # ── Step 6: Optimizer step ─────────────────────────────────────
        # Adam uses the computed gradients to update every weight:
        #   w ← w - lr × (adam_adjusted_gradient)
        # This is the actual "learning" — weights shift to reduce the loss.
        optimizer.step()

        total_loss += loss.item()   # .item() extracts Python float from tensor

    # Return mean loss per batch across this entire epoch
    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION: Evaluate on Held-Out Frames
# ─────────────────────────────────────────────────────────────────────────────

def validate(model, loader, criterion,
             frame_w: int = 1920,
             frame_h: int = 1080):
    """
    Evaluates the model on the VALIDATION set — NO weight updates occur here.

    Purpose:
      Training loss measures how well the model fits the TRAINING data.
      Validation loss measures how well it GENERALISES to unseen frames.
      If training loss keeps falling but validation loss stops improving,
      the model is over-fitting (memorising, not learning).

    Args:
        model     : AgentShockCNN (in eval mode)
        loader    : validation DataLoader
        criterion : loss function (same as training — nn.MSELoss)
        frame_w, frame_h : resolution for pixel error conversion

    Returns:
        avg_val_loss (float) — mean validation MSE loss
        avg_px_err   (float) — mean pixel error (more interpretable metric)
    """

    # model.eval() switches to INFERENCE MODE:
    #   • Dropout is DISABLED (all neurons active → deterministic output)
    #   • BatchNorm uses running mean/std (not batch statistics)
    # This is identical to what Tutorial 08 did before running FasterRCNN.
    model.eval()

    total_loss   = 0.0
    total_px_err = 0.0

    # torch.no_grad() disables gradient tracking for everything in its block.
    # We don't need gradients for validation (no backward pass), so:
    #   • Saves memory (no computation graph stored)
    #   • Speeds up inference (~30% faster in practice)
    with torch.no_grad():
        for frames, coords in tqdm(loader, desc="  Val  ", leave=False):
            frames = frames.to(DEVICE)
            coords = coords.to(DEVICE)

            preds = model(frames)

            loss          = criterion(preds, coords)
            total_loss   += loss.item()
            total_px_err += compute_pixel_error(preds, coords, frame_w, frame_h)

    avg_loss   = total_loss   / len(loader)
    avg_px_err = total_px_err / len(loader)
    return avg_loss, avg_px_err


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def train():
    """
    Main function — orchestrates the full offline training phase.

    Flow:
      build_dataloaders → instantiate model → define loss + optimizer
      → loop epochs: train_one_epoch → validate → checkpoint → early stop
    """
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # ── Step 1: Build DataLoaders ──────────────────────────────────────
    # Returns two DataLoader objects:
    #   train_loader: shuffled training batches (85% of data by default)
    #   val_loader  : ordered validation batches (15% of data)
    train_loader, val_loader = build_dataloaders(
        csv_path   = CONFIG["csv_path"],
        frames_dir = CONFIG["frames_dir"],
        batch_size = CONFIG["batch_size"],
    )

    # ── Step 2: Instantiate model ──────────────────────────────────────
    # AgentShockCNN() creates the network with randomly initialised weights.
    # .to(DEVICE) moves ALL model parameters (weights, biases, buffers) to GPU.
    # Every input tensor must also be .to(DEVICE) to match.
    model = AgentShockCNN(CONFIG["input_h"], CONFIG["input_w"]).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model parameters: {total_params:,}")

    # ── Step 3: Loss function ──────────────────────────────────────────
    # nn.MSELoss() computes:  mean( (predictions - targets)² )
    # For coordinate regression this is the standard choice.
    # It lives on CPU by default but handles GPU tensors automatically.
    criterion = nn.MSELoss()

    # ── Step 4: Optimiser ──────────────────────────────────────────────
    # optim.Adam (Adaptive Moment Estimation):
    #   Maintains a per-parameter learning rate, adapted using estimates
    #   of the first moment (mean gradient) and second moment (gradient variance).
    #   In practice: Adam converges faster and more reliably than plain SGD
    #   for most deep learning tasks.
    #
    # Parameters:
    #   model.parameters() — iterator over ALL learnable weights in the model
    #   lr                 — global learning rate (Adam adjusts per-parameter)
    #   weight_decay       — L2 regularisation: adds λ×||w||² to the loss
    #                        discourages large weights → better generalisation
    optimizer = optim.Adam(
        model.parameters(),
        lr           = CONFIG["lr"],
        weight_decay = CONFIG["weight_decay"]
    )

    # ── Step 5: Learning rate scheduler ───────────────────────────────
    # ReduceLROnPlateau: monitors validation loss and reduces lr when it
    # stops improving.
    #   mode="min"  : we want the loss to GO DOWN
    #   factor=0.5  : new_lr = old_lr × 0.5  (halve the learning rate)
    #   patience=4  : wait 4 epochs of no improvement before reducing
    #
    # Why reduce the learning rate?
    #   Early training: large steps find the rough loss minimum quickly.
    #   Later training: smaller steps fine-tune into the precise minimum
    #   without oscillating around it.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, #verbose=True
    )

    # ── Step 6: Training loop ──────────────────────────────────────────
    best_val_loss    = float("inf")  # track best model for checkpointing
    patience_counter = 0             # counts epochs without val improvement

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")

        # ── Train for one full epoch ───────────────────────────────────
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)

        # ── Evaluate on held-out validation frames ─────────────────────
        val_loss, px_err = validate(model, val_loader, criterion)

        # ── Update learning rate scheduler ────────────────────────────
        # scheduler.step() checks if val_loss improved; if not for `patience`
        # epochs it reduces the learning rate by `factor`.
        scheduler.step(val_loss)

        # ── Print epoch summary ────────────────────────────────────────
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Train Loss : {train_loss:.6f} | "
              f"Val Loss : {val_loss:.6f} | "
              f"Pixel Error : {px_err:.2f}px | "
              f"LR : {current_lr:.2e}")

        # ── Checkpoint: save best model ───────────────────────────────
        # torch.save() serialises the model's state_dict (a dict mapping
        # layer names → weight tensors) to disk. We save the entire CONFIG
        # alongside so inference.py knows the input resolution etc.
        #
        # state_dict() vs saving the whole model:
        #   torch.save(model) ties you to the exact file structure at save time.
        #   torch.save(model.state_dict()) is portable — you can rebuild the
        #   model class and load weights separately. This is best practice.
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0

            save_path = os.path.join(CONFIG["checkpoint_dir"], "best_cnn.pth")
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),  # all learned weights/biases
                "val_loss"   : val_loss,
                "px_err"     : px_err,
                "config"     : CONFIG,              # save hyperparams with model
            }, save_path)
            print(f"  ✓ Saved best model → {save_path}  "
                  f"(val_loss improved: {val_loss:.6f})")

        else:
            patience_counter += 1
            print(f"  No improvement — patience {patience_counter}/{CONFIG['patience']}")

        # ── Early stopping ─────────────────────────────────────────────
        # If validation loss hasn't improved in `patience` epochs, stop.
        # This prevents:
        #   • Wasted compute on extra epochs that won't help
        #   • Over-fitting (model starts memorising training noise)
        if patience_counter >= CONFIG["patience"]:
            print(f"\n[Early Stopping] No improvement for {CONFIG['patience']} "
                  f"consecutive epochs. Stopping at epoch {epoch}.")
            break

    print(f"\n[train] Complete. Best validation loss: {best_val_loss:.6f}")
    print(f"[train] Best model saved → {CONFIG['checkpoint_dir']}best_cnn.pth")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run the full offline training phase:
    #   python train.py
    train()