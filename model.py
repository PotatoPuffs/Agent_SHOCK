"""
model.py — CNN Architecture for Agent Shock
============================================
"The Eyes" of the Agent Shock system.

WHAT THIS FILE DOES:
  Defines the AgentShockCNN — a custom Convolutional Neural Network that
  takes a raw game frame (screenshot of Aiming.Pro) and outputs the
  NORMALISED PIXEL COORDINATES of both the crosshair and the target:

      Input  : (batch_size, 3, H, W)   — RGB frame tensor
      Output : (batch_size, 4)         — [Cx_norm, Cy_norm, Tx_norm, Ty_norm]

  This is a REGRESSION task (predict exact coordinates), NOT a classification
  task (cat vs dog). That distinction drives every architectural choice below.

HOW IT FITS INTO AGENT SHOCK:
  ┌─────────────┐    ┌─────────────────┐    ┌──────────────┐
  │  Game Frame │───▶│ AgentShockCNN   │───▶│  Δx = Tx-Cx  │───▶ RL Agent
  │ (screenshot)│    │ (this file)     │    │  Δy = Ty-Cy  │     → EMS pulse
  └─────────────┘    └─────────────────┘    └──────────────┘

KEY LIBRARY: torch.nn
  torch.nn.Module  — base class for ALL neural network models in PyTorch.
                     You subclass it and implement __init__ and forward().
  torch.nn.Sequential — chains multiple layers so data flows through each
                         in the order they are listed.
  torch.nn.Conv2d  — applies a 2D convolution filter across an image.
  torch.nn.Linear  — standard fully-connected (dense) matrix multiply layer.
  torch.nn.Sigmoid — squashes output values into (0, 1) — perfect for
                     normalised coordinate output.
"""

import numpy as np
import torch
import torch.nn as nn   # nn = neural network module — all layer types live here


# ─────────────────────────────────────────────────────────────────────────────
#  AgentShockCNN — Main Model Class
# ─────────────────────────────────────────────────────────────────────────────

class AgentShockCNN(nn.Module):
    """
    Custom CNN for coordinate regression on Aiming.Pro game frames.

    Why a CUSTOM CNN (not ResNet / VGG)?
      Pre-trained models are trained on ImageNet photos (cats, cars, people).
      Our task is geometrically simple — detect two coloured objects and
      return their centres. A lightweight custom CNN converges faster, runs
      in real-time, and doesn't need millions of ImageNet parameters.

    Architecture: two stages
      1. Feature Extraction  — convolutional blocks scan the frame spatially
      2. Position Regression — fully-connected layers map features → coordinates

    Tutorial link:
      This follows the same pattern as Tutorial 08 where torchvision models
      (FasterRCNN, FCN) each have a feature backbone followed by a task head.
      Here WE define both the backbone (self.features) and head (self.regressor).
    """

    def __init__(self, input_height=224, input_width=224):
        """
        __init__: called once when you write  model = AgentShockCNN()
        Registers all layers as PyTorch submodules so their parameters are
        tracked for gradient computation during training.

        Args:
            input_height (int): frame height fed to the model (default 224)
            input_width  (int): frame width  fed to the model (default 224)
        """

        # MUST call super().__init__() first — initialises nn.Module internals
        # (parameter tracking, device management, etc.)
        super(AgentShockCNN, self).__init__()


        # ─────────────────────────────────────────────────────────────
        # STAGE 1 — FEATURE EXTRACTION (Convolutional Backbone)
        # ─────────────────────────────────────────────────────────────
        #
        # What is a convolution?
        #   A small filter (e.g. 3×3 weights) slides across the entire image.
        #   At each position it computes a dot product with the local patch.
        #   The result is a "feature map" — bright where the filter pattern
        #   matches the image content, dark where it doesn't.
        #
        # nn.Conv2d(in_channels, out_channels, kernel_size, padding)
        #   in_channels  : how many channels enter this layer
        #                  (3 = RGB for the very first layer)
        #   out_channels : how many different filters to learn
        #                  (= number of output feature maps)
        #   kernel_size  : size of the sliding filter window — 3 = 3×3 pixels
        #   padding=1    : adds a 1-pixel border of zeros so that the spatial
        #                  size (H, W) is PRESERVED through the conv layer
        #
        # nn.BatchNorm2d(num_features)
        #   Normalises each feature map's activations to have mean≈0, std≈1.
        #   This stabilises training — gradients don't vanish or explode.
        #   In Tutorial 08 you saw how transforms normalise pixel values;
        #   BatchNorm does the same thing INSIDE the network, dynamically.
        #
        # nn.ReLU(inplace=True)
        #   Activation function: f(x) = max(0, x)
        #   Introduces non-linearity — without activations every layer is just
        #   a matrix multiply, and stacking them collapses to a single layer.
        #   inplace=True: modifies the tensor in-memory (saves a copy → faster).
        #
        # nn.MaxPool2d(kernel_size=2, stride=2)
        #   Slides a 2×2 window across the feature map and keeps ONLY the
        #   maximum value in each window.
        #   Effect: halves both H and W  →  (H, W) → (H/2, W/2)
        #   Why?  Reduces computation, creates spatial invariance (the model
        #   still detects a red blob whether it's at pixel 100 or 102).

        self.features = nn.Sequential(

            # ── Block 1: Detect low-level edges and colour boundaries ──────
            # The very first conv layer sees raw RGB pixels.
            # It learns filters like "horizontal edge", "red region boundary".
            # Input shape : (B,  3, H,   W  )
            # Output shape: (B, 16, H/2, W/2)  — after MaxPool
            nn.Conv2d(in_channels=3,  out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 224×224 → 112×112

            # ── Block 2: Detect mid-level shapes ──────────────────────────
            # Combinations of edges form shapes: the circular red target blob,
            # the + lines of the crosshair.
            # Input shape : (B, 16, H/2, W/2)
            # Output shape: (B, 32, H/4, W/4)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 112×112 → 56×56

            # ── Block 3: Detect high-level spatial structure ───────────────
            # Where in the frame is the target? Where is the crosshair?
            # These feature maps encode "there is a target-like thing at
            # this spatial position in the map".
            # Input shape : (B, 32, H/4, W/4)
            # Output shape: (B, 64, H/8, W/8)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 56×56 → 28×28
        )
        # After three MaxPool layers:  224 → 112 → 56 → 28
        # Final feature volume shape:  (B, 64, 28, 28)


        # ── Automatically compute flat size via a dummy forward pass ───────
        # Rather than manually computing 64 × 28 × 28 = 50,176, we let
        # PyTorch tell us. If you change the blocks above, this still works.
        with torch.no_grad():                              # no gradient needed
            dummy = torch.zeros(1, 3, input_height, input_width)
            dummy_out = self.features(dummy)               # run through backbone
            self._flat_size = int(np.prod(dummy_out.shape[1:]))  # e.g. 50176

        print(f"[AgentShockCNN] Flattened feature size: {self._flat_size}")


        # ─────────────────────────────────────────────────────────────
        # STAGE 2 — POSITION REGRESSION (Fully-Connected Head)
        # ─────────────────────────────────────────────────────────────
        #
        # After the convolutional backbone we have a 3D feature volume
        # (channels × height × width). We flatten this to a 1D vector and
        # pass through dense layers that learn:
        #   "given these spatial features, output 4 coordinate values"
        #
        # nn.Flatten()
        #   Collapses all dimensions except batch into one:
        #   (B, 64, 28, 28) → (B, 50176)
        #
        # nn.Linear(in_features, out_features)
        #   Fully-connected layer: output = input × W^T + b
        #   Every input neuron connects to every output neuron.
        #   This is where spatial feature positions get combined into
        #   precise coordinate estimates.
        #
        # nn.Dropout(p)
        #   During TRAINING: randomly zeroes p% of neuron outputs each
        #   forward pass. Forces the network to not rely on any one neuron
        #   → prevents over-fitting on small datasets.
        #   During INFERENCE (model.eval()): dropout is automatically disabled.
        #
        # nn.Sigmoid()
        #   Maps any real number → (0, 1):  σ(x) = 1 / (1 + e^(-x))
        #   We use this as the FINAL activation so that the 4 output values
        #   are always valid normalised coordinates in [0, 1].
        #   (No sigmoid → outputs could be negative or > 1, which is invalid
        #    for coordinates that should be fractions of frame size.)

        self.regressor = nn.Sequential(
            nn.Flatten(),                                  # (B,64,28,28)→(B,50176)

            nn.Linear(self._flat_size, 256),               # big compress step
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),                             # 40% dropout — strong regularisation

            nn.Linear(256, 128),                           # further compress
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),                             # 30% dropout

            nn.Linear(128, 4),                             # final: [Cx, Cy, Tx, Ty]
            nn.Sigmoid(),                                  # constrain output to (0,1)
        )


    def forward(self, x):
        """
        forward(): defines ONE forward pass through the network.

        PyTorch calls this automatically when you write  output = model(input).
        You should NEVER call forward() directly — always use model(input).

        Data flow:
          x  → self.features  → feature maps (spatial structure detected)
             → self.regressor → 4 normalised coordinates

        Args:
            x (torch.Tensor): shape (batch_size, 3, H, W)
                              Pixel values normalised by INFERENCE_TRANSFORM /
                              TRAIN_TRANSFORMS (ImageNet mean/std subtracted).

        Returns:
            torch.Tensor: shape (batch_size, 4)
                          Columns = [Cx_norm, Cy_norm, Tx_norm, Ty_norm]
                          All values in (0, 1).
                          Multiply by (FRAME_W, FRAME_H) to get pixel coords.

        Example (from inference.py):
            output = model(frame_tensor)    # (1, 4) — batch of 1
            cx = output[0, 0] * FRAME_W     # crosshair x in pixels
            tx = output[0, 2] * FRAME_W     # target    x in pixels
            delta_x = tx - cx               # signed error → RL agent
        """
        x = self.features(x)      # (B,3,224,224) → (B,64,28,28)
        x = self.regressor(x)     # (B,64,28,28)  → (B,4)
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  Quick sanity check — run this file directly to verify shapes
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Instantiate model (no GPU needed for a shape check)
    model = AgentShockCNN(input_height=224, input_width=224)

    # Count total learnable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Simulate a batch of 4 frames — random values stand in for real images
    dummy_batch = torch.randn(4, 3, 224, 224)  # (B=4, C=3, H=224, W=224)

    # Forward pass
    output = model(dummy_batch)

    # Verify output shape and value range
    print(f"Input  shape : {dummy_batch.shape}")   # (4, 3, 224, 224)
    print(f"Output shape : {output.shape}")         # (4, 4)
    print(f"Output min   : {output.min().item():.4f}")   # should be > 0
    print(f"Output max   : {output.max().item():.4f}")   # should be < 1
    print("Shape check PASSED ✓" if output.shape == (4, 4) else "Shape check FAILED ✗")