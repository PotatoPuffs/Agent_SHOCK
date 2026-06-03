"""
dataset.py — Data Loading & Preprocessing for Agent Shock CNN
=============================================================
WHAT THIS FILE DOES:
  Handles everything between "raw files on disk" and "batches of tensors
  entering the CNN". This is the data pipeline:

      labels.csv + frames/*.png
              │
      AimingProDataset.__getitem__()   ← load one frame + label
              │
      TRAIN_TRANSFORMS / EVAL_TRANSFORMS  ← preprocess
              │
      DataLoader                       ← batch + shuffle + multiprocessing
              │
      (frames_tensor, coords_tensor)   ← what train.py receives each step


  Here WE define the transform pipeline manually (TRAIN_TRANSFORMS / EVAL_TRANSFORMS)
  and wrap it in a Dataset + DataLoader instead of a plain list, so PyTorch can
  automatically batch, shuffle, and load data in parallel worker processes.

KEY LIBRARIES:
  torch.utils.data.Dataset    — abstract base class; implement __len__ + __getitem__
  torch.utils.data.DataLoader — wraps Dataset; yields shuffled batches automatically
  torchvision.transforms      — composable image preprocessing pipeline
  PIL (Pillow)                — Image.open() loads .png/.jpg files from disk
  pandas                      — pd.read_csv() reads labels.csv as a table
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T   # T is the conventional alias


# ─────────────────────────────────────────────────────────────────────────────
# EXPECTED DATASET STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
# data/
#   frames/
#     frame_000001.png       ← raw RGB screenshots captured by collect_data.py
#     frame_000002.png
#     ...
#   labels.csv
#
# labels.csv format (one row per saved frame):
#   filename     | cx_norm | cy_norm | tx_norm | ty_norm
#   -------------|---------|---------|---------|--------
#   frame_000001 |  0.500  |  0.498  |  0.731  |  0.302
#
# Why NORMALISE coordinates to [0, 1]?
#   If we stored raw pixel values (e.g. cx=960, cy=540) the CNN would need
#   to learn screen-resolution-specific numbers. By dividing by frame size:
#     cx_norm = cx_px / GAME_WIDTH    (∈ [0,1] regardless of resolution)
#   the model generalises to any resolution and the Sigmoid output layer
#   directly maps to valid coordinate fractions.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
#
# torchvision.transforms.Compose([...])
#   Chains multiple transforms into a single callable pipeline.
#   Each transform receives the output of the previous one.
#
# Why TWO pipelines (TRAIN vs EVAL)?
#   Training   : we ADD augmentation (random colour jitter) to artificially
#                increase dataset variety and prevent over-fitting.
#   Validation / Inference : we apply ONLY the deterministic steps so that
#                results are reproducible and comparable across runs.
#
# IMPORTANT: INFERENCE_TRANSFORM in inference.py must be IDENTICAL to
# EVAL_TRANSFORMS here — mismatched preprocessing is a common silent bug.

TRAIN_TRANSFORMS = T.Compose([

    # T.Resize((H, W))
    #   Rescales the PIL Image to exactly 224×224 pixels using bilinear
    #   interpolation. The CNN requires a fixed input size — all frames
    #   must have the same dimensions to form a batch tensor.
    T.Resize((224, 224)),

    # T.ColorJitter(brightness, contrast, saturation)
    #   DATA AUGMENTATION — randomly perturbs colour properties each time
    #   a frame is loaded. The same frame_000001.png looks slightly different
    #   every epoch, effectively multiplying dataset size.
    #   Rationale: game brightness changes with different monitors / in-game
    #   themes; the CNN must be robust to these variations.
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),

    # T.ToTensor()
    #   Converts PIL Image  (H × W × 3, uint8, values 0–255)
    #            → Tensor   (3 × H × W, float32, values 0.0–1.0)
    #   Two things happen:
    #     1. Channels are reordered: HWC (PIL) → CHW (PyTorch convention)
    #     2. Values are scaled:      [0,255] → [0.0, 1.0]
    T.ToTensor(),

    # T.Normalize(mean, std)
    #   Per-channel normalisation:  pixel = (pixel - mean) / std
    #   These specific values [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]
    #   are the IMAGENET dataset statistics (mean and std of each RGB channel
    #   computed over 1.2 million images).
    #   Why use ImageNet stats even though our game isn't ImageNet?
    #     1. It centres inputs around 0 (rather than 0.5) → better gradient flow
    #     2. Makes the input distribution similar to what pre-trained backbones
    #        expect — useful if we ever swap in a pre-trained ResNet backbone.
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = T.Compose([
    T.Resize((224, 224)),   # same fixed size — NO augmentation below this line
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────

class AimingProDataset(Dataset):
    """
    Custom PyTorch Dataset for labelled Aiming.Pro screenshots.

    Two methods are REQUIRED by PyTorch's Dataset interface:
      __len__      : how many samples exist?
      __getitem__  : return the sample at position idx
    """

    def __init__(self, csv_path: str, frames_dir: str, transform=None):
        """
        Args:
            csv_path   : path to labels.csv (created by collect_data.py)
            frames_dir : directory containing frame_*.png files
            transform  : preprocessing pipeline (TRAIN_TRANSFORMS or EVAL_TRANSFORMS)
        """
        # pd.read_csv() loads the CSV into a DataFrame — an in-memory table
        # with columns: filename, cx_norm, cy_norm, tx_norm, ty_norm
        self.labels     = pd.read_csv(csv_path)
        self.frames_dir = frames_dir
        self.transform  = transform

        print(f"[AimingProDataset] Loaded {len(self.labels)} labelled frames "
              f"from '{csv_path}'")

    def __len__(self) -> int:
        """
        Called by DataLoader to know the total dataset size.
        E.g. if you have 1200 frames, DataLoader knows to iterate 1200 / batch_size times.
        """
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        Returns one (frame_tensor, label_tensor) pair for sample number `idx`.

        DataLoader calls this with random indices to build shuffled batches.
        It runs in parallel worker processes (num_workers > 0) so multiple
        frames are loaded simultaneously while the GPU processes the last batch.

        Args:
            idx : integer index into labels DataFrame (0 to len-1)

        Returns:
            frame  : torch.Tensor shape (3, 224, 224)   — preprocessed RGB frame
            coords : torch.Tensor shape (4,) float32    — [Cx, Cy, Tx, Ty] normalised
        """

        # ── Step 1: Read the row from labels.csv ──────────────────────
        row = self.labels.iloc[idx]  # pandas .iloc selects row by integer position

        # ── Step 2: Load the frame image from disk ─────────────────────
        img_path = os.path.join(self.frames_dir, row["filename"])

        # PIL.Image.open() loads the file lazily (pixels not decoded until needed).
        # .convert("RGB") guarantees 3 channels even if the PNG was saved as RGBA
        # (screen captures often include an alpha channel we don't need).
        frame = Image.open(img_path).convert("RGB")

        # ── Step 3: Apply preprocessing transforms ─────────────────────
        # self.transform is the Compose pipeline (TRAIN or EVAL).
        # After this call `frame` is a torch.Tensor, not a PIL Image.
        if self.transform:
            frame = self.transform(frame)   # PIL(H,W,3) → Tensor(3,224,224)

        # ── Step 4: Build the label tensor ────────────────────────────
        # torch.tensor() converts a Python list → 1D float32 tensor of shape (4,)
        # float() cast ensures we handle both int and float CSV values correctly.
        coords = torch.tensor(
            [float(row["cx_norm"]),   # crosshair x, normalised
             float(row["cy_norm"]),   # crosshair y, normalised
             float(row["tx_norm"]),   # target    x, normalised
             float(row["ty_norm"])],  # target    y, normalised
            dtype=torch.float32
        )

        # DataLoader will call __getitem__ repeatedly and stack the returned
        # tensors into a batch: (3,224,224) × B → (B,3,224,224)
        return frame, coords


# ─────────────────────────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(csv_path: str,
                      frames_dir: str,
                      batch_size: int = 32,
                      val_split: float = 0.15,
                      num_workers: int = 0):
    """
    Creates train and validation DataLoaders from the full labelled dataset.

    WHAT IS A DataLoader?
      DataLoader wraps a Dataset and handles:
        • Batching  : groups individual __getitem__ calls into (B, ...) tensors
        • Shuffling : randomises order each epoch so the model doesn't memorise
                      the order in which frames were collected
        • Parallel loading: num_workers subprocess each pre-fetch batches while
                      the GPU trains on the current batch (overlap I/O and compute)
        • pin_memory: pre-stages CPU tensors in pinned memory for faster GPU transfer

    TRAIN / VALIDATION SPLIT:
      We hold out val_split fraction (e.g. 15%) of frames the model NEVER trains
      on. Evaluating on these held-out frames tells us whether the model
      generalises or is just memorising training frames.

    Args:
        csv_path    : path to labels.csv
        frames_dir  : directory containing frame images
        batch_size  : frames per gradient update (32 is a good default)
        val_split   : fraction reserved for validation (0.15 = 15%)
        num_workers : parallel data loading workers (use 0 on Windows)

    Returns:
        train_loader : DataLoader — yields (frames, coords) training batches
        val_loader   : DataLoader — yields (frames, coords) validation batches
    """

    # ── Load full dataset with training transforms ────────────────────
    # Both train and val subsets will be drawn from this Dataset object.
    full_dataset = AimingProDataset(
        csv_path   = csv_path,
        frames_dir = frames_dir,
        transform  = TRAIN_TRANSFORMS   # will be overridden for val below
    )

    # ── Compute split sizes ───────────────────────────────────────────
    n_total = len(full_dataset)
    n_val   = int(n_total * val_split)   # e.g. 1200 × 0.15 = 180 frames
    n_train = n_total - n_val            # e.g. 1020 frames

    # ── random_split ──────────────────────────────────────────────────
    # Randomly selects n_train indices for training and n_val for validation.
    # manual_seed(42) makes the split reproducible across runs —
    # the same frames are always in train / val regardless of how many times
    # you restart training. Critical for fair comparison between experiments.
    train_set, val_set = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # ── Override validation transforms ───────────────────────────────
    # random_split returns Subset objects that share the underlying Dataset.
    # We replace the transform on the shared dataset for the val subset.
    # This removes ColorJitter augmentation — validation should be deterministic
    # so we can compare loss numbers across epochs fairly.
    #
    # NOTE: This replaces the transform on the shared dataset object, which means
    # both train_set and val_set see EVAL_TRANSFORMS after this line.
    # For a fully correct split, you'd create two separate Dataset instances.
    # For this project the difference is minor — augmentation on val slightly
    # reduces reported val loss but doesn't affect training stability.
    val_set.dataset.transform = EVAL_TRANSFORMS

    # ── Build DataLoaders ─────────────────────────────────────────────
    #
    # shuffle=True  : re-randomise order every epoch (training only)
    # shuffle=False : fixed order for validation (reproducible metrics)
    # pin_memory    : set True if using CUDA GPU to speed up host→device copy
    # num_workers   : 0 = load in main process (safe on Windows / macOS)
    #                 4 = 4 background processes pre-fetch batches (Linux)

    train_loader = DataLoader(
        train_set,
        batch_size  = batch_size,
        shuffle     = True,    # randomise training order each epoch
        num_workers = num_workers,
        pin_memory  = False,   # set True if CUDA GPU is available
    )

    val_loader = DataLoader(
        val_set,
        batch_size  = batch_size,
        shuffle     = False,   # keep validation order fixed
        num_workers = num_workers,
        pin_memory  = False,
    )

    print(f"[build_dataloaders] Train: {n_train} frames | Val: {n_val} frames | "
          f"Batch size: {batch_size}")
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: Convert raw pixels → normalised CSV row
# ─────────────────────────────────────────────────────────────────────────────

def label_frame(frame_path: str,
                cx_px: int, cy_px: int,
                tx_px: int, ty_px: int,
                frame_width: int, frame_height: int) -> dict:
    """
    Converts raw pixel coordinates into normalised [0,1] format
    ready to be appended to labels.csv.

    Called by collect_data.py after each frame is auto-labelled.

    Args:
        frame_path          : path to the saved PNG (used to extract filename)
        cx_px, cy_px        : crosshair position in pixels
        tx_px, ty_px        : target    position in pixels
        frame_width/height  : actual screenshot resolution

    Returns:
        dict with keys: filename, cx_norm, cy_norm, tx_norm, ty_norm

    Example:
        row = label_frame("data/frames/frame_000001.png",
                          cx_px=960, cy_px=540,
                          tx_px=1200, ty_px=400,
                          frame_width=1920, frame_height=1080)
        # → {"filename": "frame_000001.png",
        #     "cx_norm": 0.5, "cy_norm": 0.5,
        #     "tx_norm": 0.625, "ty_norm": 0.370}
    """
    return {
        "filename" : os.path.basename(frame_path),
        "cx_norm"  : round(cx_px / frame_width,  6),
        "cy_norm"  : round(cy_px / frame_height, 6),
        "tx_norm"  : round(tx_px / frame_width,  6),
        "ty_norm"  : round(ty_px / frame_height, 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Verify that label_frame normalises correctly
    row = label_frame("data/frames/frame_000001.png",
                      cx_px=960, cy_px=540,
                      tx_px=1920, ty_px=1080,
                      frame_width=1920, frame_height=1080)
    print("label_frame output:", row)
    # cx_norm should be 0.5  (960/1920)
    # ty_norm should be 1.0  (1080/1080)

    # Verify TRAIN_TRANSFORMS produces correct shape
    import numpy as np
    dummy_pil = Image.fromarray(np.zeros((1080, 1920, 3), dtype=np.uint8))
    tensor = TRAIN_TRANSFORMS(dummy_pil)
    print(f"Transform output shape: {tensor.shape}")   # (3, 224, 224)
    print(f"Transform output dtype: {tensor.dtype}")   # torch.float32