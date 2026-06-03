"""
vision_hsv.py — HSV detection for running RL agent and EMS on live game without CNN.

Single source of truth for the colour-masking logic used to locate the
RED target sphere and the GREEN crosshair in a captured game frame.

The detection here mirrors collect_data.py, but is
generalised so the frame centre is derived from the array shape rather than
hard-coded constants — this makes it resolution-agnostic and reusable by the
live observer (integration/simulators.py::HSVBasedObserver).

OpenCV HSV scale: H∈[0,179], S∈[0,255], V∈[0,255].
Red wraps the 0°/360° hue boundary, so two ranges are OR'd together.
"""

import numpy as np
import cv2  # pip install opencv-python

# ── RED target HSV ranges ─────────────────────────────────────────────────────
TARGET_LOWER1 = np.array([0,   100, 100])   # lower red range start
TARGET_UPPER1 = np.array([15,  255, 255])   # lower red range end
TARGET_LOWER2 = np.array([160, 100, 100])   # upper red range start
TARGET_UPPER2 = np.array([180, 255, 255])   # upper red range end

TARGET_MIN_AREA = 50   # px² — ignore blobs smaller than this (noise)

# ── Crosshair (green/yellow) HSV range ────────────────────────────────────────
CROSS_LOWER = np.array([40, 100, 100])      # yellow-green
CROSS_UPPER = np.array([80, 255, 255])      # pure green
CROSSHAIR_SEARCH_RADIUS = 30                # px around centre to search

# ── Capture region ────────────────────────────────────────────────────────────
# Browser game window coordinates (for aiming.pro or similar game running in browser)
# Set these to the pixel coordinates of the game window on screen.
# top / left: top-left corner of the game area
# width / height: size of the capture region
#
# To find your values:
#   1. Open the game in browser (e.g., aiming.pro in Chrome)
#   2. Run: python -m pyautogui, which will print your mouse position in real time
#   3. Hover over top-left corner of game area → note position
#   4. Hover over bottom-right corner → note position
#   5. Set the top/left/width/height accordingly in DEFAULT_REGION below
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_REGION = {"top": 0, "left": 0, "width": 1920, "height": 1200}

def find_target(hsv_frame: np.ndarray):
    """
    Find the red target NEAREST to the frame centre via HSV colour masking.

    Multiple targets may be visible; the active one (the one being aimed at)
    is the closest to the crosshair, which sits at the centre in this drill.

    Args:
        hsv_frame : game frame in HSV colour space (H×W×3 uint8).

    Returns:
        (tx, ty) — target centroid in frame pixels, or None if not found.
    """
    h, w = hsv_frame.shape[:2]
    cx_centre = w // 2
    cy_centre = h // 2

    # Threshold both red hue ranges and OR them together
    mask1 = cv2.inRange(hsv_frame, TARGET_LOWER1, TARGET_UPPER1)
    mask2 = cv2.inRange(hsv_frame, TARGET_LOWER2, TARGET_UPPER2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # Clean up noise: open (remove specks) then close (fill holes)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = [c for c in contours if cv2.contourArea(c) > TARGET_MIN_AREA]
    if not contours:
        return None

    def dist_to_centre(contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return float("inf")
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return ((cx - cx_centre) ** 2 + (cy - cy_centre) ** 2) ** 0.5

    nearest = min(contours, key=dist_to_centre)

    M = cv2.moments(nearest)
    if M["m00"] == 0:
        return None

    tx = int(M["m10"] / M["m00"])
    ty = int(M["m01"] / M["m00"])
    return tx, ty


def find_crosshair(bgr_frame: np.ndarray):
    """
    Find the crosshair (+) position. In drill #52502 it is pinned near the
    centre, so we search a small region around centre for green pixels and
    fall back to the exact centre if none are found.

    Args:
        bgr_frame : game frame in BGR colour space (H×W×3 uint8).

    Returns:
        (cx, cy) — crosshair position in frame pixels (never None).
    """
    h, w = bgr_frame.shape[:2]
    cx_centre = w // 2
    cy_centre = h // 2

    r  = CROSSHAIR_SEARCH_RADIUS
    y1 = max(0, cy_centre - r)
    y2 = min(h, cy_centre + r)
    x1 = max(0, cx_centre - r)
    x2 = min(w, cx_centre + r)
    region = bgr_frame[y1:y2, x1:x2]

    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    cross_mask = cv2.inRange(hsv_region, CROSS_LOWER, CROSS_UPPER)

    contours, _ = cv2.findContours(cross_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            local_cx = int(M["m10"] / M["m00"])
            local_cy = int(M["m01"] / M["m00"])
            return x1 + local_cx, y1 + local_cy

    # Fallback — crosshair barely moves from centre in this drill
    return cx_centre, cy_centre