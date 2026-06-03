"""
collect_data.py — Auto-labeller for Aiming.Pro drill #52502
============================================================
WHAT THIS FILE DOES:
  Implements the first step of the OFFLINE training phase:
      [Collect Frames] → Label Coordinates → Train CNN → Validate → Save model

  Runs in the background while you play Aiming.Pro. Every 1/fps seconds it:
    1. Captures a screenshot of the game viewport (mss)
    2. Detects the RED target sphere using HSV colour masking (OpenCV)
    3. Detects the crosshair position (centre or green-pixel search)
    4. Saves the frame as a PNG and records normalised coordinates to CSV

  Here we AUTOMATICALLY generate a labelled dataset:
    • Targets are bright RED spheres on a dark blue background
      → reliably detectable via HSV colour threshold (no manual labelling)
    • Crosshair is always near screen centre in this drill
      → fixed-region search or geometric fallback

KEY LIBRARIES:
  mss      — screen capture (same as inference.py)
  cv2      — OpenCV: image processing, HSV conversion, contour detection
  PIL      — screenshot → numpy conversion
  numpy    — array operations for colour masking
  csv      — writing labels.csv row by row
"""

import os
import time
import csv
import numpy as np
from PIL import Image
import mss
import cv2   # pip install opencv-python


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT PATHS
# ─────────────────────────────────────────────────────────────────────────────
SAVE_DIR   = "data/frames/"   # PNG frames saved here
LABELS_CSV = "data/labels.csv"
os.makedirs(SAVE_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN / CAPTURE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
FRAME_W  = 1920    # full screen width
FRAME_H  = 1200    # full screen height (adjust to your monitor)

# Game viewport — exclude browser chrome (tabs bar, bookmarks, address bar)
# The game content starts at approximately y=110 pixels from screen top.
GAME_TOP    = 110
GAME_LEFT   = 0
GAME_WIDTH  = 1920
GAME_HEIGHT = 1090   # FRAME_H - GAME_TOP


# ─────────────────────────────────────────────────────────────────────────────
# RED TARGET: HSV COLOUR RANGE
# ─────────────────────────────────────────────────────────────────────────────
# OpenCV HSV scale: H∈[0,179], S∈[0,255], V∈[0,255]
# (OpenCV halves the standard 360° hue to fit in uint8)
#
# Red wraps around 0°/360° on the hue wheel, need TWO ranges:
#   Lower red range: H ∈ [0, 15]    (0° → orange-red)
#   Upper red range: H ∈ [160, 179] (320° → magenta-red)
# We OR the two masks together to capture all shades of red.

TARGET_LOWER1 = np.array([0,   100, 100])   # lower red range start
TARGET_UPPER1 = np.array([15,  255, 255])   # lower red range end
TARGET_LOWER2 = np.array([160, 100, 100])   # upper red range start
TARGET_UPPER2 = np.array([180, 255, 255])   # upper red range end


# ─────────────────────────────────────────────────────────────────────────────
# CROSSHAIR SEARCH PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# The crosshair (+) in drill #52502 is always near the centre of the viewport.
# We search within a small 60×60 pixel region around the centre for a
# green/yellow pixel cluster (the crosshair colour from the screenshots).
CROSSHAIR_SEARCH_RADIUS = 30   # pixels around centre to search


# ─────────────────────────────────────────────────────────────────────────────
# TARGET DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def find_target(hsv_frame: np.ndarray):
    """
    Finds the NEAREST red target to the screen centre using HSV colour masking.

    WHY NEAREST?
      Multiple red targets appear on screen simultaneously. The target that
      matters for training is the one closest to the crosshair (the one the
      player should currently be aiming at).

    How it works:
      1. cv2.inRange() creates a binary mask: 255 where pixel ∈ [lower,upper],
         0 everywhere else.
      2. Two masks (for the two red hue ranges) are OR'd together.
      3. Morphological operations remove tiny noise blobs.
      4. cv2.findContours() finds the outlines of connected white regions.
      5. We filter by area (>50px²) to ignore noise.
      6. The contour whose centroid is closest to screen centre is returned.

    Args:
        hsv_frame : the game frame in HSV colour space (numpy array H×W×3)

    Returns:
        (tx, ty) — target centroid in game-frame pixels
        None     — if no red blob found
    """
    # Step 1: Threshold — create binary masks for both red hue ranges
    mask1 = cv2.inRange(hsv_frame, TARGET_LOWER1, TARGET_UPPER1)
    mask2 = cv2.inRange(hsv_frame, TARGET_LOWER2, TARGET_UPPER2)
    mask  = cv2.bitwise_or(mask1, mask2)   # combine: any red pixel → white

    # Step 2: Morphological operations to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    # MORPH_OPEN  = erode then dilate: removes small isolated noise pixels
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    # MORPH_CLOSE = dilate then erode: fills small holes inside detected blobs
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Step 3: Find contours of red blobs
    # cv2.findContours returns (contours, hierarchy)
    # RETR_EXTERNAL: only outer contours (no nested ones)
    # CHAIN_APPROX_SIMPLE: compress horizontal/vertical/diagonal segments → fewer points
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Step 4: Filter tiny blobs (noise artefacts) — real targets are large spheres
    contours = [c for c in contours if cv2.contourArea(c) > 50]
    if not contours:
        return None

    # Step 5: Find the blob closest to the screen centre (= active target)
    cx_centre = GAME_WIDTH  // 2
    cy_centre = GAME_HEIGHT // 2

    def dist_to_centre(contour):
        """Euclidean distance from contour centroid to game viewport centre."""
        M = cv2.moments(contour)   # image moments → statistical properties
        if M["m00"] == 0:          # m00 = area; 0 means degenerate contour
            return float("inf")
        cx = M["m10"] / M["m00"]  # centroid x = first moment x / area
        cy = M["m01"] / M["m00"]  # centroid y = first moment y / area
        return ((cx - cx_centre)**2 + (cy - cy_centre)**2) ** 0.5

    nearest = min(contours, key=dist_to_centre)

    # Step 6: Compute centroid of the nearest (active) target blob
    M  = cv2.moments(nearest)
    if M["m00"] == 0:
        return None

    tx = int(M["m10"] / M["m00"])
    ty = int(M["m01"] / M["m00"])
    return tx, ty


# ─────────────────────────────────────────────────────────────────────────────
# CROSSHAIR DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def find_crosshair(game_frame_bgr: np.ndarray):
    """
    Finds the crosshair (+) position in the game frame.

    Process:
      The crosshair is a small bright green/white + symbol always near the
      centre of the game viewport in drill #52502. We:
        1. Crop a small 60×60 region around the viewport centre.
        2. Search for green/yellow HSV pixels (crosshair colour from screenshots).
        3. If found, return the centroid of the green pixels.
        4. If not found, fall back to exact screen centre (safe default).

    Args:
        game_frame_bgr : game viewport in BGR colour space (numpy array)

    Returns:
        (cx, cy) — crosshair position in game-frame pixels
    """
    h, w = game_frame_bgr.shape[:2]
    cx_centre = w // 2
    cy_centre = h // 2

    # Step 1: Crop small search region around centre
    r  = CROSSHAIR_SEARCH_RADIUS
    y1 = max(0, cy_centre - r)
    y2 = min(h, cy_centre + r)
    x1 = max(0, cx_centre - r)
    x2 = min(w, cx_centre + r)
    region = game_frame_bgr[y1:y2, x1:x2]

    # Step 2: Convert region to HSV and threshold for green/yellow crosshair
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Green HSV range: H∈[40,80] covers yellow-green through pure green
    cross_lower = np.array([40,  100, 100])
    cross_upper = np.array([80,  255, 255])
    cross_mask  = cv2.inRange(hsv_region, cross_lower, cross_upper)

    # Step 3: Find contours of green pixels in the search region
    contours, _ = cv2.findContours(cross_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Found green pixels — use the largest cluster's centroid
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            local_cx = int(M["m10"] / M["m00"])
            local_cy = int(M["m01"] / M["m00"])
            # Convert from search-region coordinates back to full game-frame coordinates
            return x1 + local_cx, y1 + local_cy

    # Step 4: Fallback — crosshair is always very close to centre in this drill
    # This is accurate enough for training since the crosshair barely moves.
    return cx_centre, cy_centre


# ─────────────────────────────────────────────────────────────────────────────
# MAIN COLLECTION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def collect(duration_seconds: int = 240, fps: int = 10):
    """
    Main data collection loop — runs for `duration_seconds` capturing frames.

    At each tick (1/fps seconds):
      1. Capture game viewport as RGB screenshot
      2. Convert to BGR (OpenCV convention) and HSV (for colour detection)
      3. Detect red target — skip frame if not found
      4. Detect crosshair position
      5. Save PNG frame to disk
      6. Append normalised coordinates to in-memory list
    At the end: flush all rows to labels.csv (append mode — safe to re-run).

    Args:
        duration_seconds : how long to collect (recommend 2–10 minutes)
        fps              : capture rate in frames per second
                           10fps × 120s = ~1200 frames before filtering
    """
    print(f"[collect] Auto-collecting for {duration_seconds}s at {fps}fps")
    print("Switch to your Aiming.Pro browser tab NOW — collection starts in 3s\n")
    time.sleep(3)   # give you time to alt-tab to the browser

    frame_interval = 1.0 / fps

    # mss capture region — game viewport only (excludes browser chrome)
    region = {
        "top"   : GAME_TOP,
        "left"  : GAME_LEFT,
        "width" : GAME_WIDTH,
        "height": GAME_HEIGHT,
    }

    rows      = []    # accumulates CSV row dicts
    frame_idx = 0     # count of saved (labelled) frames
    skipped   = 0     # count of skipped frames (no target detected)
    end_time  = time.time() + duration_seconds

    with mss.MSS() as sct:
        while time.time() < end_time:
            t0 = time.perf_counter()   # start of this frame's time budget

            # ── Step 1: Capture game viewport ─────────────────────────
            screenshot = sct.grab(region)
            img_rgb = Image.frombytes("RGB", screenshot.size,
                                       screenshot.bgra, "raw", "BGRX")

            # ── Step 2: Convert colour spaces for OpenCV processing ────
            # OpenCV expects BGR (not RGB) — convert once and cache
            img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
            # HSV used for both target and crosshair detection
            hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            # ── Step 3: Detect target ──────────────────────────────────
            target = find_target(hsv)
            if target is None:
                # No red blob found — skip this frame (don't save unlabelled data)
                skipped += 1
                elapsed = time.perf_counter() - t0
                if frame_interval - elapsed > 0:
                    time.sleep(frame_interval - elapsed)
                continue   # skip to next frame

            tx, ty = target   # target pixel position in game-frame coordinates

            # ── Step 4: Detect crosshair ───────────────────────────────
            cx, cy = find_crosshair(img_bgr)   # crosshair pixel position

            # ── Step 5: Save frame PNG ─────────────────────────────────
            filename = f"frame_{frame_idx:06d}.png"
            img_rgb.save(os.path.join(SAVE_DIR, filename))

            # ── Step 6: Record normalised label ───────────────────────
            # Divide by game viewport dimensions (NOT full screen) so that
            # coordinates are relative to the cropped capture region —
            # consistent with what the CNN will see during inference.
            rows.append({
                "filename" : filename,
                "cx_norm"  : round(cx / GAME_WIDTH,  6),
                "cy_norm"  : round(cy / GAME_HEIGHT, 6),
                "tx_norm"  : round(tx / GAME_WIDTH,  6),
                "ty_norm"  : round(ty / GAME_HEIGHT, 6),
            })

            frame_idx += 1

            # Live status line (overwrites itself with \r)
            print(f"  Frame {frame_idx:04d} | "
                  f"target=({tx:4d},{ty:4d})  "
                  f"crosshair=({cx:4d},{cy:4d}) | "
                  f"skipped={skipped}", end="\r")

            # ── Frame rate control ────────────────────────────────────
            elapsed = time.perf_counter() - t0
            if frame_interval - elapsed > 0:
                time.sleep(frame_interval - elapsed)

    # ── Write all rows to labels.csv ──────────────────────────────────
    # Append mode: re-running collect() adds MORE frames to the same CSV.
    # write_header only if the file is new/empty.
    write_header = not os.path.exists(LABELS_CSV) or \
                   os.path.getsize(LABELS_CSV) == 0

    with open(LABELS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "cx_norm", "cy_norm", "tx_norm", "ty_norm"]
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"\n\n[collect] Done!")
    print(f"  Labelled frames saved : {frame_idx}")
    print(f"  Frames skipped        : {skipped}  (no red target detected)")
    print(f"  CSV path              : {LABELS_CSV}")
    print(f"  Frames directory      : {SAVE_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Collect 4 minutes of gameplay at 10fps ≈ 1200 raw frames
    # Skipped frames reduce this; a typical run yields 600–900 labelled frames.
    # Run collect() multiple times (sessions append to the same CSV).
    collect(duration_seconds=240, fps=10)