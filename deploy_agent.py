"""
deploy_agent.py — Live deployment against real aiming.pro game.

Pipeline each frame:
    CNN observation  →  RL agent  →  EMS action

No simulation. No AimingEnv. The browser game is the environment.
"""

import argparse
import time
import numpy as np
import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

from stable_baselines3 import PPO

# Swap stubs for real implementations when teammates are ready
from integration.simulators import SimulatedCNNObserver as CNNObserver
from integration.simulators import SimulatedEMSController as EMSController
# from perception.cnn_observer import CNNObserver
# from hardware.ems_controller import EMSController

# ACTION_MAP duplicated here so deploy has no dependency on AimingEnv
ACTION_MAP = {
    # 0: ("none"),
    0: ("left"),
    1: ("right"),
    2: ("click"),
}

SCREEN_W = 1280
STEP_PX = 12 # adapt to take in average movement from EMS controller test data
TARGET_RADIUS = 30
CLICK_Y = 130 + 720 // 2   # vertical centre of capture region

def run(args):
    print("\n=== Agent SHOCK — Deployment ===")
    print(f"EMS: {'dry-run' if args.dry_run else 'live'}\n")

    model = PPO.load("models/aiming_ppo")
    observer = CNNObserver(screen_w=SCREEN_W)
    ems = EMSController()

    cursor_x = SCREEN_W / 2
    frame_count = 0
    hits = 0

    try:
        while True:
            t0 = time.perf_counter()

            # 1. CNN observation
            obs, target_x = observer.build_obs(cursor_x)

            # 2. RL agent picks action
            action, _ = model.predict(obs, deterministic=True)
            action_ = ACTION_MAP[int(action)]

           # 3. Update virtual cursor
            if action_ == "left":
                cursor_x = max(0, cursor_x - STEP_PX)
            elif action_ == "right":
                cursor_x = min(SCREEN_W, cursor_x + STEP_PX)

            # 4. Send to EMS first — includes click signal
            ems.send_action(action_)

            # 5. Handle click consequences
            if action_ == "click":
                if not args.dry_run:
                    pyautogui.click(int(cursor_x), CLICK_Y)
                if abs(cursor_x - target_x) < TARGET_RADIUS:
                    hits += 1
                ems.reset()

            

            frame_count += 1
            elapsed = time.perf_counter() - t0

            print(
                f"\r  frame={frame_count:5d}  "
                f"target={target_x:5.0f}px  "
                f"cursor={cursor_x:5.0f}px  "
                f"Δx={abs(cursor_x - target_x):4.0f}px  "
                f"action={action_:5s}  "
                f"hits={hits}  "
                f"fps={1/elapsed if elapsed > 0 else 0:.0f}  ",
                end="", flush=True,
            )

            time.sleep(max(0, 1/30 - elapsed))

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        ems.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="No real clicks — EMS still fires (or stub prints)")
    args = parser.parse_args()
    run(args)