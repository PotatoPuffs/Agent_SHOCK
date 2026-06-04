"""
cnn_observer.py — CNN observer for the Agent SHOCK pipeline (self-contained).

One file, four jobs:
    python cnn_observer.py --collect --seconds 180   # build HSV-labelled dataset
    python cnn_observer.py --train                    # train CNN on the CSV
    python cnn_observer.py --eval                     # held-out pixel-error metrics
    # ...and import RealCNNObserver for `run.py --cnn real`

LABEL SOURCE
    HSV-as-labeller. While playing, vision_hsv.find_target / find_crosshair
    auto-labels each captured frame. The CNN learns to reproduce those
    coordinates from raw pixels, so at inference it needs no colour masking.

CONTRACT (must match HSVBasedObserver exactly so the RL policy transfers)
    get_state(last_dx, pulse_duration_ms) -> (obs, target_x, cursor_x)
        obs = [norm_error, norm_cursor_x, last_dx_norm, pulse_dur_norm]  float32
        target_x, cursor_x are in SCREEN_W (contract) pixel space.

NO TRAIN/INFER SKEW
    preprocess() is the single image pipeline used by BOTH the dataset and the
    live observer. Change it in one place or not at all.
"""

import os
import csv
import time
import argparse
import numpy as np

# ── Capture region — laptop default; change top/left/width/height freely ──────
CAPTURE_REGION = {"top": 0, "left": 0, "width": 1920, "height": 1200}

# ── Model input size (keep 16:10 to match 1920x1200; small = fast) ────────────
IMG_W = 160
IMG_H = 100

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "data_cnn"
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
MODEL_PT   = "models/cnn_observer2.pt"

# ── Contract constants (fallback values if interfacing.py isn't importable) ───
try:
    from integration.interfacing import (
        BaseCNNObserver, SCREEN_W, TARGET_RADIUS, MAX_DX,
    )
except Exception:                      # allows standalone collect/train without the package
    SCREEN_W      = 1280
    TARGET_RADIUS = 30
    MAX_DX        = 60


# ── Shared preprocessing — used by dataset AND observer (no skew) ─────────────
def preprocess(bgr_frame: np.ndarray) -> np.ndarray:
    """
    BGR uint8 (H×W×3) -> float32 CHW in [0,1] at IMG_H×IMG_W, RGB order.
    Returns a numpy array shaped (3, IMG_H, IMG_W).
    """
    import cv2
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    chw = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    return chw


# ── Capture helper (mss) ──────────────────────────────────────────────────────
def _grab_bgr(sct, region):
    import cv2
    shot = sct.grab(region)
    bgra = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


# ── 1. COLLECT — HSV auto-labelled dataset ────────────────────────────────────
def collect(seconds: int = 180, fps: int = 10):
    """
    Capture frames while you play; label each with vision_hsv; save PNG + CSV row.
    Coordinates are stored NORMALISED to the capture region (resolution-agnostic):
        tx_norm, ty_norm, cx_norm, cy_norm  in [0,1]
    Frames with no detected target are skipped (no usable label).
    """
    import cv2
    import mss
    from vision_hsv import find_target, find_crosshair

    os.makedirs(FRAMES_DIR, exist_ok=True)
    cap_w = float(CAPTURE_REGION["width"])
    cap_h = float(CAPTURE_REGION["height"])

    print(f"[collect] {seconds}s @ {fps}fps, region={CAPTURE_REGION}")
    print("[collect] Switch to the game window now — starting in 3s...")
    time.sleep(3)

    interval = 1.0 / fps
    rows, idx, skipped = [], 0, 0
    end = time.time() + seconds

    with mss.mss() as sct:
        while time.time() < end:
            t0 = time.perf_counter()
            bgr = _grab_bgr(sct, CAPTURE_REGION)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            target = find_target(hsv)
            if target is None:                       # no label -> skip frame
                skipped += 1
            else:
                tx, ty = target
                cx, cy = find_crosshair(bgr)
                fname  = f"f_{idx:06d}.png"
                cv2.imwrite(os.path.join(FRAMES_DIR, fname), bgr)
                rows.append({
                    "filename": fname,
                    "tx_norm": round(tx / cap_w, 6), "ty_norm": round(ty / cap_h, 6),
                    "cx_norm": round(cx / cap_w, 6), "cy_norm": round(cy / cap_h, 6),
                })
                idx += 1
                print(f"\r[collect] saved={idx}  skipped={skipped}  "
                      f"target=({tx},{ty}) cross=({cx},{cy})", end="", flush=True)

            dt = time.perf_counter() - t0
            if interval - dt > 0:
                time.sleep(interval - dt)

    write_header = (not os.path.exists(LABELS_CSV)) or os.path.getsize(LABELS_CSV) == 0
    with open(LABELS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "tx_norm", "ty_norm", "cx_norm", "cy_norm"])
        if write_header:
            w.writeheader()
        w.writerows(rows)

    print(f"\n[collect] done. saved={idx} skipped={skipped} -> {LABELS_CSV}")


# ── 2/3. Dataset + Model (torch imported lazily so --collect needs no torch) ──
def _build_torch_bits():
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    import cv2

    class AimingDataset(Dataset):
        """Reads labels.csv, applies the shared preprocess(), returns (img, [tx,ty,cx,cy])."""
        def __init__(self, csv_path, frames_dir, rows=None):
            self.frames_dir = frames_dir
            if rows is not None:
                self.rows = rows
            else:
                with open(csv_path) as f:
                    self.rows = list(csv.DictReader(f))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r   = self.rows[i]
            bgr = cv2.imread(os.path.join(self.frames_dir, r["filename"]))
            x   = torch.from_numpy(preprocess(bgr))            # (3,H,W) float32
            y   = torch.tensor([float(r["tx_norm"]), float(r["ty_norm"]),
                                float(r["cx_norm"]), float(r["cy_norm"])], dtype=torch.float32)
            return x, y

    class CoordCNN(nn.Module):
        """Tiny conv regressor -> 4 sigmoid outputs (tx,ty,cx,cy) in [0,1]."""
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),  # ->50x80
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # ->25x40
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # ->12x20
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 4), nn.Sigmoid(),
            )

        def forward(self, x):
            return self.head(self.features(x))

    return torch, nn, AimingDataset, CoordCNN


def _split_rows(val_frac=0.2, seed=0):
    with open(LABELS_CSV) as f:
        rows = list(csv.DictReader(f))
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac))
    return rows[n_val:], rows[:n_val]      # train, val


# ── 2. TRAIN ──────────────────────────────────────────────────────────────────
def train(epochs=50, batch_size=32, lr=1e-3, val_frac=0.2):
    torch, nn, AimingDataset, CoordCNN = _build_torch_bits()
    from torch.utils.data import DataLoader

    if not os.path.exists(LABELS_CSV):
        raise SystemExit(f"No dataset at {LABELS_CSV}. Run --collect first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_rows, val_rows = _split_rows(val_frac)
    print(f"[train] device={device}  train={len(train_rows)}  val={len(val_rows)}  "
          f"epochs={epochs}  input={IMG_W}x{IMG_H}")

    tr = DataLoader(AimingDataset(LABELS_CSV, FRAMES_DIR, train_rows),
                    batch_size=batch_size, shuffle=True,  num_workers=2)
    va = DataLoader(AimingDataset(LABELS_CSV, FRAMES_DIR, val_rows),
                    batch_size=batch_size, shuffle=False, num_workers=2)

    model = CoordCNN().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train(); tr_loss = 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward(); opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(tr.dataset)

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(device), y.to(device)
                va_loss += lossf(model(x), y).item() * x.size(0)
        va_loss /= len(va.dataset)
        print(f"  ep {ep:3d}/{epochs}  train_mse={tr_loss:.5f}  val_mse={va_loss:.5f}")

    os.makedirs(os.path.dirname(MODEL_PT), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PT)
    print(f"[train] saved -> {MODEL_PT}")


# ── 3. EVALUATE — pixel-error metrics on held-out val set ─────────────────────
def evaluate_cnn(val_frac=0.2):
    torch, nn, AimingDataset, CoordCNN = _build_torch_bits()
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val_rows = _split_rows(val_frac)
    va = DataLoader(AimingDataset(LABELS_CSV, FRAMES_DIR, val_rows),
                    batch_size=64, shuffle=False)

    model = CoordCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PT, map_location=device))
    model.eval()

    cap_w, cap_h = CAPTURE_REGION["width"], CAPTURE_REGION["height"]
    t_err, c_err = [], []                          # per-frame euclidean pixel error
    with torch.no_grad():
        for x, y in va:
            p = model(x.to(device)).cpu().numpy()
            y = y.numpy()
            for pi, yi in zip(p, y):
                tdx = (pi[0] - yi[0]) * cap_w; tdy = (pi[1] - yi[1]) * cap_h
                cdx = (pi[2] - yi[2]) * cap_w; cdy = (pi[3] - yi[3]) * cap_h
                t_err.append((tdx**2 + tdy**2) ** 0.5)
                c_err.append((cdx**2 + cdy**2) ** 0.5)

    t_err, c_err = np.array(t_err), np.array(c_err)
    within = float(np.mean(t_err < TARGET_RADIUS))
    print(f"\n=== CNN held-out eval ({len(val_rows)} frames, capture px) ===")
    print(f"  Target  mean={t_err.mean():6.1f}px  p90={np.percentile(t_err,90):6.1f}px")
    print(f"  Cursor  mean={c_err.mean():6.1f}px  p90={np.percentile(c_err,90):6.1f}px")
    print(f"  Target within TARGET_RADIUS ({TARGET_RADIUS}px capture-equiv): {within*100:.1f}%")
    print("  (interpretation: CNN reproduces HSV labels to within these px.)")


# ── 4. LIVE OBSERVER — drop-in for run.py --cnn real ──────────────────────────
class RealCNNObserver(BaseCNNObserver):
    """
    Live CNN observer. Same get_state() contract as HSVBasedObserver:
    detected positions are scaled from capture-pixel space into SCREEN_W space.
    """
    def __init__(self, screen_w: int = SCREEN_W, capture_region: dict = None,
                 model_path: str = MODEL_PT, detect_crosshair: bool = True):
        import torch
        import mss
        _, _, _, CoordCNN = _build_torch_bits()

        self.screen_w         = screen_w
        self.region           = dict(capture_region or CAPTURE_REGION)
        self.detect_crosshair = detect_crosshair
        self._cap_w           = float(self.region["width"])
        self._scale           = screen_w / self._cap_w     # capture-px -> contract-px

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch  = torch
        self._model  = CoordCNN().to(self._device)
        self._model.load_state_dict(torch.load(model_path, map_location=self._device))
        self._model.eval()
        self._sct = mss.mss()
        print(f"[CNNObserver] active — device={self._device}, region={self.region}, "
              f"scale capture {int(self._cap_w)}px -> contract {screen_w}px.")

    def get_state(self, last_dx: float, pulse_duration_ms: float):
        bgr = _grab_bgr(self._sct, self.region)
        x   = self._torch.from_numpy(preprocess(bgr)).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            tx_n, ty_n, cx_n, cy_n = self._model(x)[0].cpu().numpy()

        target_x = float(tx_n * self._cap_w * self._scale)             # -> contract px
        cursor_x = (float(cx_n * self._cap_w * self._scale)
                    if self.detect_crosshair else self.screen_w / 2.0)

        norm_error   = (cursor_x - target_x) / self.screen_w
        norm_cursor  = cursor_x / self.screen_w
        last_dx_norm = float(np.clip(last_dx / MAX_DX, -1.0, 1.0))
        pulse_norm   = pulse_duration_ms / 1000.0

        obs = np.array([norm_error, norm_cursor, last_dx_norm, pulse_norm], dtype=np.float32)
        return obs, target_x, cursor_x

    def close(self):
        try:
            self._sct.close()
        except Exception:
            pass


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CNN observer — collect / train / eval")
    ap.add_argument("--collect", action="store_true", help="capture HSV-labelled frames")
    ap.add_argument("--train",   action="store_true", help="train CNN on labels.csv")
    ap.add_argument("--eval",    action="store_true", help="held-out pixel-error metrics")
    ap.add_argument("--seconds", type=int, default=180)
    ap.add_argument("--fps",     type=int, default=10)
    ap.add_argument("--epochs",  type=int, default=25)
    args = ap.parse_args()

    if args.collect:
        collect(seconds=args.seconds, fps=args.fps)
    elif args.train:
        train(epochs=args.epochs)
    elif args.eval:
        evaluate_cnn()
    else:
        ap.print_help()