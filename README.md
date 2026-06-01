# Agent Shock — CNN Perception Subsystem
### Aim-Assistance AI Using CNN + Reinforcement Learning + EMS

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [CNN — Purpose, Objective and Process](#3-cnn--purpose-objective-and-process)
4. [CNN Architecture](#4-cnn-architecture)
5. [Important Parameters and Features](#5-important-parameters-and-features)
6. [File Overview](#6-file-overview)
7. [Prerequisites](#7-prerequisites)
8. [Installation and Setup](#8-installation-and-setup)
9. [Step-by-Step Commands and Expected Outputs](#9-step-by-step-commands-and-expected-outputs)
10. [Understanding the Output Metrics](#10-understanding-the-output-metrics)
11. [Troubleshooting](#11-troubleshooting)
12. [Common Mistakes and How to Avoid Them](#12-common-mistakes-and-how-to-avoid-them)
13. [Quick Reference](#13-quick-reference)

---

## 1. Project Overview

Agent Shock is a closed-loop AI aim-assistance system. A Convolutional Neural Network (CNN) acts as the "eyes" of the system — it watches the game screen in real time, detects where the crosshair and target are, and computes the error between them. That error is passed to a Reinforcement Learning (RL) agent which decides how strongly to fire an EMS (Electrical Muscle Stimulation) pulse through a TENS machine, physically contracting the user's forearm muscles to move the mouse towards the target.

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT SHOCK LOOP                         │
│                                                                 │
│  Game Screen  →  CNN  →  Δx / Δy  →  RL Agent  →  EMS Pulse   │
│      ↑                                                  │       │
│      └─────────── mouse moves ──────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

**This repository covers the CNN component only** — the perception subsystem that reads the screen and outputs coordinate errors.

---

## 2. System Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                    OFFLINE — Training Phase                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌──────────────────┐     ┌──────────────────┐                  ║
║  │  collect_data.py  │────▶│  data/frames/    │                  ║
║  │                   │     │  data/labels.csv  │                  ║
║  │  mss screen cap   │     └────────┬─────────┘                  ║
║  │  HSV colour mask  │              │                            ║
║  │  Auto-labelling   │              ▼                            ║
║  └──────────────────┘     ┌──────────────────┐                  ║
║                            │   dataset.py     │                  ║
║                            │  Transforms      │                  ║
║                            │  DataLoaders     │                  ║
║                            │  Train/Val split │                  ║
║                            └────────┬─────────┘                  ║
║                                     │                            ║
║                                     ▼                            ║
║                            ┌──────────────────┐                  ║
║                            │    train.py      │                  ║
║                            │  MSELoss         │                  ║
║                            │  Adam optimiser  │                  ║
║                            │  Early stopping  │                  ║
║                            └────────┬─────────┘                  ║
║                                     │                            ║
║                                     ▼                            ║
║                            ┌──────────────────┐                  ║
║                            │  best_cnn.pth    │ ← saved weights  ║
║                            └──────────────────┘                  ║
╠══════════════════════════════════════════════════════════════════╣
║                    LIVE — Inference Phase                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Game Screen                                                     ║
║      │                                                           ║
║      ▼                                                           ║
║  capture_frame()     ← mss reads OS display buffer              ║
║      │                                                           ║
║      ▼                                                           ║
║  preprocess_frame()  ← resize 224×224, normalise                ║
║      │                                                           ║
║      ▼                                                           ║
║  AgentShockCNN()     ← load best_cnn.pth weights                ║
║  forward pass        → [Cx, Cy, Tx, Ty] normalised              ║
║      │                                                           ║
║      ▼                                                           ║
║  Δx = Tx - Cx        ← signed horizontal pixel error            ║
║  Δy = Ty - Cy        ← signed vertical pixel error              ║
║      │                                                           ║
║      ▼                                                           ║
║  game_state dict     ← positions + distances + movement history ║
║      │                                                           ║
║      ▼                                                           ║
║  rl_agent_callback   → action (0=none, 1=low, 2=med, 3=high)   ║
║      │                                                           ║
║      ▼                                                           ║
║  micro-ROS → ESP32 → TENS relay → forearm contraction           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 3. CNN — Purpose, Objective and Process

### Purpose

The CNN solves a **visual perception problem**: given a raw screenshot of the game, determine the exact pixel position of two objects — the crosshair and the target.

This is not a classification task ("is this a target?") — it is a **regression task** ("where exactly is the target?"). The output is four continuous coordinate values, not a category label.

### Objective

Minimise the difference between the CNN's predicted coordinates and the true coordinates across all training frames. This is measured using **Mean Squared Error (MSE)**:

```
Loss = mean( (predicted_coord - true_coord)² )
```

Squaring the error means large misses are penalised disproportionately — a 100px error costs 100x more than a 10px error. This is important because large misses would cause the EMS to fire in the wrong direction entirely.

### The Full Process

```
OFFLINE PHASE

Step 1 — Collect
  Play Aiming.Pro with collect_data.py running in background.
  Every 0.1 seconds:
    • Screenshot captured via mss
    • Red target found via HSV colour masking (OpenCV)
    • Crosshair found near screen centre
    • Frame saved as PNG
    • Coordinates normalised and written to labels.csv

Step 2 — Train
  train.py reads labels.csv and frames/
  For each epoch (pass through all training data):
    • DataLoader yields batches of 32 frames
    • CNN predicts [Cx, Cy, Tx, Ty] for each frame
    • MSELoss measures how wrong predictions are
    • loss.backward() computes which weights caused the error
    • Adam optimizer nudges weights to reduce the loss
  Best model (lowest val loss) saved to checkpoints/best_cnn.pth

Step 3 — Validate
  validate_localisation.py loads best_cnn.pth
  Reports per-object accuracy:
    • Crosshair pixel error (mean, median, max)
    • Target pixel error (mean, median, max)
    • % of frames within 10/20/30/50px threshold

LIVE PHASE

Step 4 — Inference
  inference.py loads best_cnn.pth
  Runs continuously while game is open:
    • Captures screen every ~33ms (30 FPS target)
    • Preprocesses frame identically to training
    • CNN forward pass → [Cx, Cy, Tx, Ty]
    • Computes Δx, Δy, Euclidean distance
    • Tracks previous frame movement
    • Passes game_state to rl_agent_callback
```

### Why Normalise Coordinates?

Raw pixel coordinates (e.g. cx=960, cy=545) are resolution-dependent. By dividing by frame dimensions:

```
cx_norm = cx_px / frame_width     → value in [0, 1]
cy_norm = cy_px / frame_height    → value in [0, 1]
```

The model works at any resolution, and the final Sigmoid activation layer naturally outputs values in (0, 1) — matching the label format exactly.

### What Δx and Δy Mean

```
Δx = Target_x − Crosshair_x     (horizontal signed error)
Δy = Target_y − Crosshair_y     (vertical signed error)

Δx > 0  →  target is to the RIGHT  →  EMS fires rightward pulse
Δx < 0  →  target is to the LEFT   →  EMS fires leftward pulse
Δx ≈ 0  →  crosshair is on target  →  no EMS needed

Δy > 0  →  target is BELOW crosshair
Δy < 0  →  target is ABOVE crosshair
(vertical EMS not currently implemented)
```

---

## 4. CNN Architecture

```
Input: (batch_size, 3, 224, 224)   ← batch of RGB game frames

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1 — Feature Extraction (Convolutional Backbone)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Block 1 — Detect edges and colour boundaries
  Conv2d(3→16, 3×3)  → BatchNorm → ReLU
  Conv2d(16→16, 3×3) → BatchNorm → ReLU
  MaxPool2d(2×2)
  Output: (B, 16, 112, 112)

Block 2 — Detect shapes (target circle, crosshair lines)
  Conv2d(16→32, 3×3) → BatchNorm → ReLU
  Conv2d(32→32, 3×3) → BatchNorm → ReLU
  MaxPool2d(2×2)
  Output: (B, 32, 56, 56)

Block 3 — Detect spatial layout (where are the objects?)
  Conv2d(32→64, 3×3) → BatchNorm → ReLU
  Conv2d(64→64, 3×3) → BatchNorm → ReLU
  MaxPool2d(2×2)
  Output: (B, 64, 28, 28)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 2 — Position Regression (Fully Connected Head)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Flatten:              (B, 64, 28, 28) → (B, 50176)
  Linear(50176 → 256)   → ReLU → Dropout(0.4)
  Linear(256 → 128)     → ReLU → Dropout(0.3)
  Linear(128 → 4)       → Sigmoid

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output: (B, 4)   ← [Cx_norm, Cy_norm, Tx_norm, Ty_norm]
                    all values in (0, 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total parameters: ~13 million
```

### Layer-by-Layer Explanation

| Layer | What It Does |
|---|---|
| `Conv2d` | Slides a small filter across the image; learns to activate at specific visual patterns |
| `BatchNorm2d` | Normalises activations to stabilise training — same idea as normalising pixel values |
| `ReLU` | Introduces non-linearity: f(x) = max(0, x) — without this all layers collapse to one |
| `MaxPool2d(2×2)` | Halves spatial dimensions — reduces computation, creates position invariance |
| `Flatten` | Collapses 3D feature volume (64, 28, 28) → 1D vector (50176,) |
| `Linear` | Fully connected layer — maps features to coordinate estimates |
| `Dropout` | Randomly zeros neurons during training — prevents overfitting on small datasets |
| `Sigmoid` | Squashes output to (0, 1) — ensures valid normalised coordinate output |

---

## 5. Important Parameters and Features

### Training Parameters (train.py CONFIG)

| Parameter | Default | What It Controls | Guidance |
|---|---|---|---|
| `batch_size` | 32 | Frames per weight update | Lower if GPU memory error; higher = faster but needs more memory |
| `epochs` | 50 | Max training passes | Early stopping usually triggers before this |
| `lr` | 1e-3 | Learning rate — size of each weight update step | Too high = unstable; too low = slow |
| `weight_decay` | 1e-4 | L2 regularisation — penalises large weights | Helps generalisation on small datasets |
| `patience` | 8 | Epochs without improvement before early stopping | Increase if model seems to stop too early |
| `input_h/w` | 224 | CNN input resolution | Must match across all files — do not change without retraining |

### Data Collection Parameters (collect_data.py)

| Parameter | Default | What It Controls |
|---|---|---|
| `GAME_WIDTH` | 1920 | Your screen width — adjust to match your monitor |
| `GAME_HEIGHT` | 1090 | Game viewport height (screen height minus browser chrome) |
| `GAME_TOP` | 110 | Pixels from top of screen to where game content starts |
| `fps` | 10 | Capture rate — 10fps x 120s = ~1200 raw frames per session |
| `duration_seconds` | 120 | Length of each collection session |

### Inference Parameters (inference.py)

| Parameter | Default | What It Controls |
|---|---|---|
| `FRAME_W / FRAME_H` | 1920 / 1080 | Screen resolution for denormalising coordinates |
| `target_fps` | 30 | Target inference rate — GPU needed to achieve this |
| `CAPTURE_REGION` | Full screen | Crop to game window only for better performance |

### Key Features

**Auto-labelling via HSV colour masking**
The data collector automatically detects red targets using OpenCV's HSV colour thresholding — no manual labelling required. Red is detected using two hue ranges (0–15° and 160–180°) because red wraps around the HSV hue wheel.

**Weighted loss for target vs crosshair**
Because the crosshair barely moves in this drill, it dominates the loss signal. Adding target weighting forces the CNN to focus more on the harder problem:
```python
cx_loss = criterion(preds[:, :2], coords[:, :2])
tx_loss = criterion(preds[:, 2:], coords[:, 2:])
loss    = cx_loss + 3.0 * tx_loss
```

**Game state with frame history**
The inference loop maintains prev_coords and prev_action across frames so the RL agent receives not just current positions but also velocity (how fast the target is moving) and previous action context.

**Early stopping**
Training automatically stops when validation loss stops improving for patience consecutive epochs — prevents wasted compute and overfitting.

**Learning rate scheduling**
ReduceLROnPlateau halves the learning rate when validation loss plateaus — large steps early for fast convergence, small steps later for fine-tuning.

---

## 6. File Overview

| File | Phase | Purpose |
|---|---|---|
| `model.py` | Both | CNN architecture — convolutional backbone + coordinate regression head |
| `dataset.py` | Offline | Loads frames from disk, applies transforms, builds train/val DataLoaders |
| `collect_data.py` | Offline | Captures game screenshots and auto-labels target/crosshair coordinates |
| `train.py` | Offline | Full training loop — loss, optimiser, checkpointing, early stopping |
| `validate_localisation.py` | Offline | Per-object accuracy report — crosshair and target separately |
| `inference.py` | Live | Real-time screen capture → CNN → game_state → RL agent callback |
| `requirements.txt` | Setup | All Python package dependencies |

### Folder Structure After Running Everything

```
agent_shock/
├── model.py
├── dataset.py
├── collect_data.py
├── train.py
├── validate_localisation.py
├── inference.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── frames/
│   │   ├── frame_000000.png     ← collected game screenshots
│   │   └── ...
│   └── labels.csv               ← normalised coordinates per frame
│
├── checkpoints/
│   └── best_cnn.pth             ← trained model weights
│
├── localisation_results.csv     ← per-frame validation results
└── venv/                        ← virtual environment
```

---

## 7. Prerequisites

### Python Version

Python **3.9 or newer** is required.

```bash
python --version
# Expected: Python 3.9.x or higher
```

Download from https://www.python.org/downloads/ if not installed.
On Windows: tick **"Add Python to PATH"** during installation.

### GPU vs CPU

| | CPU | GPU (NVIDIA CUDA) |
|---|---|---|
| Training speed | ~2–5 min per epoch | ~10–20 sec per epoch |
| Inference FPS | ~7–8 FPS | ~30–60 FPS |
| Latency per frame | ~130ms | ~15ms |
| EMS reliability | Sluggish corrections | Real-time corrections |

The code runs on CPU — but for live inference at 30 FPS a GPU is strongly recommended.

---

## 8. Installation and Setup

### Step 1 — Create a Virtual Environment

```bash
# Inside your project folder
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS / Linux
source venv/bin/activate
```

You will see `(venv)` at the start of your terminal prompt. Always activate before running any project commands.

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Version | Why It Is Needed |
|---|---|---|
| `torch` | >= 2.0.0 | PyTorch — deep learning framework. Tensors, autograd, nn.Module, optimisers |
| `torchvision` | >= 0.15.0 | Image transforms used in dataset.py and inference.py |
| `numpy` | >= 1.24.0 | Array operations, colour mask arrays, coordinate arithmetic |
| `pandas` | >= 2.0.0 | Reads labels.csv into a DataFrame for the Dataset class |
| `Pillow` | >= 9.0.0 | Loads PNG frames from disk; converts mss screenshots to PIL Images |
| `opencv-python` | >= 4.7.0 | HSV colour thresholding, contour detection in collect_data.py |
| `mss` | >= 9.0.0 | Ultra-fast screen capture — reads OS display buffer directly |
| `tqdm` | >= 4.65.0 | Live progress bar during training epochs |

PyTorch is a large download (~2–3 GB). This may take 10–20 minutes on a slow connection.

### Step 3 — GPU Setup (Optional but Recommended)

The default install gives the CPU-only version. If you have an NVIDIA GPU:

```bash
# Uninstall CPU version first
pip uninstall torch torchvision

# Install with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Find the right command for your system at: https://pytorch.org/get-started/locally/

Verify GPU is detected:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected if GPU found:
# True
# NVIDIA GeForce RTX 3080
```

### Step 4 — Fix the mss Deprecation Warning

In `inference.py` change:
```python
# FROM:
with mss.mss() as sct:
# TO:
with mss.MSS() as sct:
```

In `collect_data.py` change:
```python
# Import FROM:
from mss import MSS
# Import TO:
import mss

# Usage FROM:
with MSS() as sct:
# Usage TO:
with mss.MSS() as sct:
```

---

## 9. Step-by-Step Commands and Expected Outputs

### Command 1 — Verify Installation

```bash
python -c "import torch, torchvision, numpy, pandas, PIL, cv2, mss, tqdm; print('All imports OK')"
```

Expected:
```
All imports OK
```

---

### Command 2 — Verify CNN Architecture

```bash
python model.py
```

Expected:
```
[AgentShockCNN] Flattened feature size: 50176
Total parameters: 12,951,252
Input  shape : torch.Size([4, 3, 224, 224])
Output shape : torch.Size([4, 4])
Output min   : 0.4227
Output max   : 0.5795
Shape check PASSED ✓
```

What each line confirms:

| Line | Meaning |
|---|---|
| `Flattened feature size: 50176` | 64 × 28 × 28 = 50,176 after three MaxPool layers |
| `Total parameters: ~13M` | Number of learnable weights in the whole network |
| `Output shape: [4, 4]` | Batch of 4 frames → 4 coordinate predictions each |
| `Output min/max in (0,1)` | Sigmoid activation is working correctly |
| `Shape check PASSED` | End-to-end forward pass completed without errors |

---

### Command 3 — Collect Training Data

Open Aiming.Pro drill #52502 in your browser before running this.

```bash
python collect_data.py
```

Expected:
```
[collect] Auto-collecting for 120s at 10fps
Switch to your Aiming.Pro browser tab NOW — collection starts in 3s

  Frame 0198 | target=( 980, 476)  crosshair=( 960, 545) | skipped=22

[collect] Done!
  Labelled frames saved : 198
  Frames skipped        : 22  (no red target detected)
  CSV path              : data/labels.csv
  Frames directory      : data/frames/
```

Tips:
- You have 3 seconds to click on your browser window after running
- Run this command multiple times — each session appends to the same CSV
- Aim for 3000–5000 total labelled frames across all sessions
- If skipped frames are very high (>50%), the game window may not be focused
- Play naturally and move the mouse to create varied crosshair/target positions

---

### Command 4 — Train the CNN

```bash
python train.py
```

Expected:
```
[AimingProDataset] Loaded 786 labelled frames from 'data/labels.csv'
[build_dataloaders] Train: 669 frames | Val: 117 frames | Batch size: 32
[AgentShockCNN] Flattened feature size: 50176
[train] Model parameters: 12,951,252
[train] Training device: cpu

Epoch 1/50
  Train Loss : 0.135058 | Val Loss : 0.014156 | Pixel Error : 105.14px | LR : 1.00e-03
  ✓ Saved best model → checkpoints/best_cnn.pth

Epoch 2/50
  Train Loss : 0.011053 | Val Loss : 0.002964 | Pixel Error : 37.31px | LR : 1.00e-03
  ✓ Saved best model → checkpoints/best_cnn.pth

...

[Early Stopping] No improvement for 8 consecutive epochs. Stopping at epoch 35.
[train] Complete. Best validation loss: 0.002858
[train] Best model saved → checkpoints/best_cnn.pth
```

What to watch for:

| Metric | Epoch 1 | After Training | Notes |
|---|---|---|---|
| Train Loss | ~0.13 | ~0.003 | Should decrease each epoch |
| Val Loss | ~0.014 | ~0.002 | Drives checkpointing and early stopping |
| Pixel Error | ~105px | ~30–50px | More interpretable than loss |
| LR | 1.00e-03 | 1.25e-04 | Scheduler halves it when plateauing |

---

### Command 5 — Validate Localisation Accuracy

```bash
python validate_localisation.py
```

Expected:
```
============================================================
  CNN LOCALISATION ACCURACY REPORT
============================================================

── Euclidean Pixel Error (straight-line distance) ──────────
  Crosshair │ Mean: 4.3px   Median: 2.1px   Max: 13.6px
  Target    │ Mean: 152.7px  Median: 76.0px  Max: 538.0px

── Horizontal Error (x-axis) ────────────────────────────────
  Crosshair │ Mean: 4.2px   Median: 2.0px
  Target    │ Mean: 100.8px  Median: 48.1px

── Vertical Error (y-axis) ──────────────────────────────────
  Crosshair │ Mean: 0.7px   Median: 0.5px
  Target    │ Mean: 78.9px  Median: 10.6px

── Frames Where Prediction Lands Within Threshold ───────────
  Threshold       Crosshair       Target
  ------------------------------------
  ≤ 10       px       100.0%         4.0%
  ≤ 20       px       100.0%        13.0%
  ≤ 30       px       100.0%        22.2%
  ≤ 50       px       100.0%        36.2%
============================================================
```

Decision — is the model ready?

| Target Mean Error | Action |
|---|---|
| > 80px | Not ready — check label quality in labels.csv, then retrain |
| 30–80px | Borderline — collect more data and retrain |
| 10–30px | Acceptable — proceed to inference |
| < 10px | Excellent — proceed to inference |

If adding more data makes accuracy worse, the auto-labeller is generating bad labels. Check labels.csv — if tx_norm values are erratic or stuck at 0.5, the target was never properly found during collection.

---

### Command 6 — Run Live Inference

Only run this once validation shows acceptable accuracy.

```bash
python inference.py
```

Expected:
```
[AgentShockCNN] Flattened feature size: 50176
[load_model] Loaded checkpoint from epoch 27 | val_loss=0.002858 | pixel_error=32.42px
[inference] Starting live loop at 30 FPS target | Device: cpu
Press Ctrl+C to stop.

[Frame 00030] Δx=  +22.0px  Δy=  -24.6px  Latency=131.4ms  FPS≈7.6
[Frame 00060] Δx=  +45.3px  Δy=  -39.0px  Latency=127.6ms  FPS≈7.8
[Frame 00090] Δx=  +23.5px  Δy=  -23.0px  Latency=125.7ms  FPS≈8.0
```

Reading the output:

| Value | Example | Meaning |
|---|---|---|
| `Δx = +22.0px` | Positive | Target is 22px to the RIGHT of crosshair |
| `Δx = -41.0px` | Negative | Target is 41px to the LEFT of crosshair |
| `Δx ≈ 0px` | Near zero | Crosshair is on target — no EMS needed |
| `Latency = 131ms` | CPU-bound | Time for one full capture→preprocess→CNN cycle |
| `FPS ≈ 7.6` | CPU-bound | GPU would push this to 30–60 FPS |

Press Ctrl+C to stop.

---

## 10. Understanding the Output Metrics

### Pixel Error vs Loss

| Metric | Unit | When Used | Interpretation |
|---|---|---|---|
| MSE Loss | Normalised (0–1) | During training | Lower is better; hard to interpret directly |
| Pixel Error | Pixels | Training + validation | Human-readable — "CNN is off by X pixels on average" |
| Euclidean Error | Pixels | Validation report | Straight-line distance between predicted and true position |

### Threshold Table

The threshold table answers: "what fraction of frames land the prediction within X pixels of the true position?"

```
≤ 10px threshold           ≤ 30px threshold

     ┌──┐                    ┌──────────┐
     │ ●│  ← 10px radius     │    ●     │  ← 30px radius
     └──┘                    └──────────┘
  Strict — few frames       Lenient — most frames qualify
  qualify
```

For the EMS system, ≤ 30px on the target x-axis is the most meaningful threshold — frames within 30px produce an EMS pulse that is at least directionally correct.

### Latency and FPS

```
Latency = time for one full loop iteration:
  capture_frame()      ~10–30ms
  preprocess_frame()    ~5–10ms
  CNN forward pass    ~80–120ms  ← bottleneck on CPU
  compute Δx/Δy         ~0.1ms

CPU total:  ~130ms  →  FPS ≈ 7–8
GPU total:   ~15ms  →  FPS ≈ 30–60
```

FPS is the control loop frequency — at 7 FPS the EMS reacts to where the target was 140ms ago. At 30 FPS it reacts to 33ms ago — far more responsive for real-time correction.

### game_state Dictionary

Every frame the inference loop builds and passes this to the RL agent:

| Key | Type | Meaning |
|---|---|---|
| `cx`, `cy` | pixels | Current crosshair position |
| `tx`, `ty` | pixels | Current target position |
| `delta_x` | pixels ± | Signed horizontal distance — drives EMS direction |
| `delta_y` | pixels ± | Signed vertical distance |
| `distance` | pixels | Straight-line gap between crosshair and target |
| `target_move_x` | pixels ± | How far target moved horizontally since last frame |
| `target_move_y` | pixels ± | How far target moved vertically since last frame |
| `prev_target_direction` | +1/0/-1 | Was target moving right, still, or left |
| `prev_cursor_direction` | +1/0/-1 | Was cursor moving right, still, or left |
| `prev_action` | 0–3 | What EMS pulse level fired last frame |

---

## 11. Troubleshooting

**`ModuleNotFoundError: No module named 'torch'`**
```bash
venv\Scripts\activate          # activate virtual environment first
pip install -r requirements.txt
```

**`collect_data.py` saves 0 frames or very high skip rate**
- The game window was not focused when collection started
- Adjust `GAME_WIDTH`, `GAME_HEIGHT`, `GAME_TOP` in collect_data.py to match your screen
- Verify the red targets are clearly visible

**`FileNotFoundError: data/labels.csv`**
Run `collect_data.py` before `train.py`.

**`FileNotFoundError: checkpoints/best_cnn.pth`**
Run `train.py` to completion before `inference.py`.

**Training loss not decreasing**
- Collect more data (aim for 3000+ frames)
- Reduce learning rate: `"lr": 1e-4` in CONFIG
- Check labels.csv has values between 0 and 1 in all columns

**Validation accuracy gets worse with more data**
The auto-labeller is producing bad labels. Check that tx_norm in labels.csv varies across rows — if it is stuck at 0.5 the target was never found during collection.

**`RuntimeError: CUDA out of memory`**
Reduce batch size: `"batch_size": 16` or `8` in CONFIG.

**`DeprecationWarning: mss.mss is deprecated`**
Replace `mss.mss()` with `mss.MSS()` in inference.py and collect_data.py. See Section 8 Step 4.

**Inference FPS too low (< 15 FPS)**
- Install GPU version of PyTorch (Section 8 Step 3)
- Set `CAPTURE_REGION` to game window only instead of full screen
- Reduce `target_fps` to 15 in `run_inference_loop(target_fps=15)`

**`game_state is not defined` in rl_agent_callback**
The callback signature is outdated. Change:
```python
# FROM:
def rl_agent_callback(delta_x: float, delta_y: float, coords: dict):
# TO:
def rl_agent_callback(game_state: dict) -> int:
```

---

## 12. Common Mistakes and How to Avoid Them

| Mistake | Consequence | How to Avoid |
|---|---|---|
| Not activating virtual environment | Packages not found | Always check for (venv) in terminal prompt |
| Running train.py before collecting enough data | Poor accuracy | Collect 3000+ frames minimum |
| Different preprocessing in inference vs training | Silent accuracy loss | Keep INFERENCE_TRANSFORM identical to EVAL_TRANSFORMS |
| Not calling model.eval() before inference | Non-deterministic predictions | Already handled in load_model() — do not remove it |
| Running inference.py with a poor model | Wrong-direction EMS pulses | Always check validate_localisation.py first |
| Changing input_h/w without retraining | Shape mismatch crash | Change CONFIG and retrain from scratch |

---

## 13. Quick Reference

### Full Command Sequence

```bash
# ── One-time setup ──────────────────────────────────
cd path/to/agent_shock
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

# ── Verify ──────────────────────────────────────────
python model.py                  # Shape check PASSED ✓

# ── Collect (repeat until 3000–5000 total frames) ───
python collect_data.py
python collect_data.py
python collect_data.py

# ── Train ────────────────────────────────────────────
python train.py                  # saves checkpoints/best_cnn.pth

# ── Validate ─────────────────────────────────────────
python validate_localisation.py  # target mean error < 30px?
# if poor → collect more data → retrain

# ── Live inference ────────────────────────────────────
python inference.py              # Ctrl+C to stop
```

### When to Re-run Each Step

| Situation | Commands to re-run |
|---|---|
| Want more training data | `collect_data.py` → `train.py` → `validate_localisation.py` |
| Accuracy is poor | `collect_data.py` → `train.py` → `validate_localisation.py` |
| Changed CONFIG in train.py | `train.py` → `validate_localisation.py` |
| Changed model architecture | `model.py` verify → `train.py` → `validate_localisation.py` |
| Fresh machine or new environment | Full setup from Section 8 |
| Just checking live output | `inference.py` only |

### Accuracy Targets

| Object | Target Mean Pixel Error | Typical Result |
|---|---|---|
| Crosshair | < 10px | Easily achieved — barely moves in this drill |
| Target | < 30px | Requires 3000+ frames and clean auto-labels |

### Key Data Formats

```
labels.csv columns:
  filename | cx_norm | cy_norm | tx_norm | ty_norm
  All coordinate values normalised to [0, 1]

best_cnn.pth contains:
  epoch | model_state | val_loss | px_err | config

game_state keys:
  cx, cy, tx, ty           ← current pixel positions
  delta_x, delta_y         ← signed errors (drive EMS)
  distance                 ← Euclidean gap
  target_move_x/y          ← target velocity
  prev_target_direction    ← +1 right / -1 left / 0 still
  prev_action              ← last EMS pulse level (0–3)
```