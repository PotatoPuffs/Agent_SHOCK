"""
validate_localisation.py
========================
Measures CNN localisation accuracy separately for:
  - Crosshair (Cx, Cy)
  - Target    (Tx, Ty)

Produces per-object pixel error, directional breakdown (x vs y),
and a worst-case frame analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from model   import AgentShockCNN
from dataset import AimingProDataset, EVAL_TRANSFORMS

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_W    = 1920
FRAME_H    = 1080
MODEL_PATH = "checkpoints/best_cnn.pth"


def evaluate_localisation(csv_path, frames_dir):
    """
    Runs the CNN over every frame in the dataset and records:
      - Crosshair predicted vs true position
      - Target    predicted vs true position
    Separately, so you can see which object the CNN struggles with.
    """

    # ── Load dataset and model ────────────────────────────────────────
    dataset = AimingProDataset(csv_path, frames_dir, transform=EVAL_TRANSFORMS)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = AgentShockCNN(224, 224).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # ── Storage for per-frame results ─────────────────────────────────
    results = []

    with torch.no_grad():
        for frames, coords in loader:
            frames = frames.to(DEVICE)
            coords = coords.to(DEVICE)
            preds  = model(frames)

            # Move to CPU numpy for analysis
            p = preds.cpu().numpy()   # (B, 4) — predicted
            t = coords.cpu().numpy()  # (B, 4) — ground truth

            # columns: [Cx_norm, Cy_norm, Tx_norm, Ty_norm]
            # Denormalise each coordinate back to pixels
            p_cx = p[:, 0] * FRAME_W;  t_cx = t[:, 0] * FRAME_W
            p_cy = p[:, 1] * FRAME_H;  t_cy = t[:, 1] * FRAME_H
            p_tx = p[:, 2] * FRAME_W;  t_tx = t[:, 2] * FRAME_W
            p_ty = p[:, 3] * FRAME_H;  t_ty = t[:, 3] * FRAME_H

            for i in range(len(p)):
                results.append({
                    # ── Crosshair errors ──────────────────────────────
                    "cx_err_x" : abs(p_cx[i] - t_cx[i]),  # horizontal error
                    "cx_err_y" : abs(p_cy[i] - t_cy[i]),  # vertical error
                    "cx_err_px": np.sqrt(                  # Euclidean distance
                        (p_cx[i] - t_cx[i])**2 +
                        (p_cy[i] - t_cy[i])**2
                    ),
                    # ── Target errors ─────────────────────────────────
                    "tx_err_x" : abs(p_tx[i] - t_tx[i]),
                    "tx_err_y" : abs(p_ty[i] - t_ty[i]),
                    "tx_err_px": np.sqrt(
                        (p_tx[i] - t_tx[i])**2 +
                        (p_ty[i] - t_ty[i])**2
                    ),
                })

    return pd.DataFrame(results)


def print_report(df):
    """
    Prints a readable accuracy report split by object and axis.
    """
    print("\n" + "="*60)
    print("  CNN LOCALISATION ACCURACY REPORT")
    print("="*60)

    # ── Per-object Euclidean pixel error ──────────────────────────────
    print("\n── Euclidean Pixel Error (straight-line distance) ──────────")
    print(f"  Crosshair │ Mean: {df['cx_err_px'].mean():.1f}px  "
          f"Median: {df['cx_err_px'].median():.1f}px  "
          f"Max: {df['cx_err_px'].max():.1f}px")
    print(f"  Target    │ Mean: {df['tx_err_px'].mean():.1f}px  "
          f"Median: {df['tx_err_px'].median():.1f}px  "
          f"Max: {df['tx_err_px'].max():.1f}px")

    # ── Directional breakdown: x vs y ─────────────────────────────────
    print("\n── Horizontal Error (x-axis) ────────────────────────────────")
    print(f"  Crosshair │ Mean: {df['cx_err_x'].mean():.1f}px  "
          f"Median: {df['cx_err_x'].median():.1f}px")
    print(f"  Target    │ Mean: {df['tx_err_x'].mean():.1f}px  "
          f"Median: {df['tx_err_x'].median():.1f}px")

    print("\n── Vertical Error (y-axis) ──────────────────────────────────")
    print(f"  Crosshair │ Mean: {df['cx_err_y'].mean():.1f}px  "
          f"Median: {df['cx_err_y'].median():.1f}px")
    print(f"  Target    │ Mean: {df['tx_err_y'].mean():.1f}px  "
          f"Median: {df['tx_err_y'].median():.1f}px")

    # ── Within-threshold accuracy (like IoU but for points) ───────────
    print("\n── Frames Where Prediction Lands Within Threshold ───────────")
    print(f"  {'Threshold':<12} {'Crosshair':>12} {'Target':>12}")
    print(f"  {'-'*36}")
    for thresh in [10, 20, 30, 50]:
        cx_pct = (df["cx_err_px"] <= thresh).mean() * 100
        tx_pct = (df["tx_err_px"] <= thresh).mean() * 100
        print(f"  ≤ {thresh:<9}px  {cx_pct:>10.1f}%  {tx_pct:>10.1f}%")

    print("\n" + "="*60)


if __name__ == "__main__":
    df = evaluate_localisation(
        csv_path   = "data/labels.csv",
        frames_dir = "data/frames/"
    )
    print_report(df)
    df.to_csv("localisation_results.csv", index=False)
    print("\nFull per-frame results saved → localisation_results.csv")