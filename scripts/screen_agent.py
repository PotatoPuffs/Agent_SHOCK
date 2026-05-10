"""
screen_agent.py — Run a trained agent against the LIVE aiming.pro browser game.

The agent watches the real screen via mss (fast screenshot lib) and uses
the trained policy to decide actions. The virtual cursor is decoupled from
the OS mouse — the agent updates an internal float, and we use pyautogui
ONLY to click (not to move the mouse). Cursor movement is delivered via
EMS in real mode, or skipped entirely in screen-only eval mode.

Usage:
    # Watch the live game, see what the agent WOULD do (no actual clicking):
    python screen_agent.py --dry-run

    # Click for real using pyautogui (no EMS):
    python screen_agent.py --click

    # Full EMS mode (connect MCU first):
    python screen_agent.py --ems --port /dev/ttyUSB0
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.aiming_env import AimingEnv

try:
    from stable_baselines3 import PPO
    import mss
    import cv2
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0
except ImportError as e:
    print(f"Missing dep: {e}")
    print("pip install mss opencv-python pyautogui stable-baselines3")
    sys.exit(1)


# ── Blue blob detector (works on default aiming.pro targets) ──────────────────

def detect_target_x(frame_bgr, debug=False):
    """Return x-centre of largest cyan/blue target, or None."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Tune these ranges if aiming.pro changes target colour
    mask = cv2.inRange(hsv,
                       np.array([85, 120, 120]),
                       np.array([135, 255, 255]))
    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:   # filter tiny noise blobs
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    if debug:
        cv2.drawContours(frame_bgr, [c], -1, (0, 255, 0), 2)
        cv2.circle(frame_bgr, (cx, int(M["m01"] / M["m00"])), 8, (0, 255, 255), -1)
        cv2.imshow("Detection debug", frame_bgr)
        cv2.waitKey(1)
    return cx


# ── Screen capture region — adjust to your browser window ────────────────────

CAPTURE_REGION = {
    "top": 130,      # pixels from top of screen to game area
    "left": 0,
    "width": 1280,
    "height": 720,
}
CROSSHAIR_X = CAPTURE_REGION["width"] // 2   # centre of game view


# ── Main agent loop ───────────────────────────────────────────────────────────

def run(args):
    print("\n=== Screen Agent — Live aiming.pro ===")
    print(f"  Mode: {'dry-run' if args.dry_run else 'EMS' if args.ems else 'click-only'}")
    print(f"  Capture: {CAPTURE_REGION}")
    print("  Press Ctrl+C to stop.\n")

    model = PPO.load("./models/aiming_ppo")

    # Internal virtual cursor (agent's belief about where it is)
    cursor_x = float(CROSSHAIR_X)
    screen_w = CAPTURE_REGION["width"]
    step_px = 12
    target_radius = 30

    sct = mss.mss()
    last_target_x = None
    last_vx = 1.0

    frame_count = 0
    hits = 0

    # Serial for EMS
    ser = None
    if args.ems:
        import serial
        ser = serial.Serial(args.port, baudrate=115200, timeout=1)
        time.sleep(2)
        print(f"  Serial open: {args.port}")

    try:
        while True:
            t0 = time.perf_counter()

            # 1. Grab frame
            raw = sct.grab(CAPTURE_REGION)
            frame = np.array(raw)[:, :, :3]
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 2. Detect target
            target_x = detect_target_x(frame_bgr, debug=args.debug)
            if target_x is None:
                target_x = last_target_x or screen_w // 2

            # Estimate target velocity
            if last_target_x is not None:
                last_vx = np.sign(target_x - last_target_x) or last_vx
            last_target_x = target_x

            # 3. Build observation
            error = (cursor_x - target_x) / screen_w
            obs = np.array([error, cursor_x / screen_w, last_vx], dtype=np.float32)

            # 4. Agent decides
            action, _ = model.predict(obs, deterministic=True)

            # 5. Execute
            pixel_error = abs(cursor_x - target_x)
            action_name = ["LEFT", "RIGHT", "CLICK"][action]

            if action == 0:
                cursor_x = max(0, cursor_x - step_px)
                if args.ems and ser:
                    ser.write(b"C0D50\n")
            elif action == 1:
                cursor_x = min(screen_w, cursor_x + step_px)
                if args.ems and ser:
                    ser.write(b"C1D50\n")
            elif action == 2:
                if not args.dry_run:
                    # Translate virtual cursor_x to screen coords and click
                    screen_click_x = CAPTURE_REGION["left"] + int(cursor_x)
                    screen_click_y = CAPTURE_REGION["top"] + CAPTURE_REGION["height"] // 2
                    pyautogui.click(screen_click_x, screen_click_y)
                if pixel_error < target_radius:
                    hits += 1

            frame_count += 1
            elapsed = time.perf_counter() - t0
            fps = 1 / elapsed if elapsed > 0 else 0

            print(
                f"\r  frame={frame_count:5d}  "
                f"target_x={target_x:4.0f}  cursor_x={cursor_x:4.0f}  "
                f"error={pixel_error:4.0f}px  "
                f"action={action_name:5s}  hits={hits}  fps={fps:.0f}   ",
                end="", flush=True
            )

            # Throttle to ~30 fps
            sleep = max(0, 1/30 - elapsed)
            time.sleep(sleep)

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        if ser:
            ser.close()
        cv2.destroyAllWindows()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Don't click, just observe")
    parser.add_argument("--click", action="store_true", help="Click with pyautogui")
    parser.add_argument("--ems", action="store_true", help="Send EMS commands over serial")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port for MCU")
    parser.add_argument("--debug", action="store_true", help="Show CV detection window")
    args = parser.parse_args()
    run(args)