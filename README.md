# AimingRL — Human-Robot Hybrid Aiming System

Train a reinforcement learning agent in simulation, then deploy via EMS to a human arm.

```
Agent_SHOCK/
├── env/
│   ├── aiming_env.py        ← Custom Gym environment (sim / screen / real modes)
│   └── ems_controller.ino   ← Arduino firmware (MCU relay control)
├── scripts/
│   ├── train.py             ← RL training loop (PPO via Stable-Baselines3)
│   └── screen_agent.py      ← Live agent against real browser game
└── requirements.txt
```

---

## Phase 1 — Install

```bash
cd Agent_SHOCK
pip install -r requirements.txt
```
Also include this repository for your own data collection for EMS: https://github.com/jspsych/jsPsych/releases.
---

## Phase 2 — Train in simulation (no screen, no mouse, no EMS)

```bash
# Fast headless training — 500k steps takes ~10–20 min on CPU
python scripts/train.py

# Watch the agent play while training (slower, needs a display)
python scripts/train.py --render

# Training produces:
#   ./models/aiming_ppo.zip      ← saved policy
#   ./training_curve.png         ← reward + hits plot, updated every 5k steps
#   ./tb_logs/                   ← TensorBoard logs
```

**What the agent learns:**
- Observation: [pixel_error, cursor_x, target_direction]  (3 numbers)
- Actions: move_left | move_right | click
- Reward: –|error|/width per step, +10 on hit, –2 on miss, –5 on timeout

The target bounces left/right randomly, just like aiming.pro's lateral-movement drills.

---

## Phase 3 — Verify against the real game (no EMS yet)

Open aiming.pro drill in your browser. Keep it in the foreground.

```bash
# Watch what the agent WOULD do — no clicking, no EMS
python scripts/screen_agent.py --dry-run --debug

# Actually click using pyautogui (agent controls clicks, human moves mouse manually)
python scripts/screen_agent.py --click
```

The `--debug` flag opens a CV window showing what the blob detector sees.
Tune `CAPTURE_REGION` in `screen_agent.py` to match your browser window position.

---

## Phase 4 — EMS deployment (real arm)

### Hardware setup
1. Upload `env/ems_controller.ino` to Arduino (Uno / Nano / Mega)
2. Wire two relay modules:
   - Relay 1 → TENS channel 1 (left muscle)
   - Relay 2 → TENS channel 2 (right muscle)
3. Set TENS intensity LOW (start at level 5, increase slowly)
4. Place electrodes on forearm — one pair per channel

### Run
```bash
python scripts/screen_agent.py --ems --port /dev/ttyUSB0
# On Windows: --port COM3
```

Serial protocol: `C<0|1>D<ms>\n`
- `C0D50\n` = fire left channel for 50ms
- `C1D50\n` = fire right channel for 50ms

---

## Tuning tips

| Parameter | File | Effect |
|---|---|---|
| `step_px` | `train.py` | Larger = faster cursor, harder to be precise |
| `target_radius` | `train.py` | Bigger = easier hits, less accuracy needed |
| `ent_coef` | `train.py` | Higher = more exploration (helpful early) |
| Pulse duration `D<ms>` | `screen_agent.py` | Longer = stronger contraction |
| `COOLDOWN_MS` | `ems_controller.ino` | Safety gap — don't reduce below 300ms |
| HSV range in `detect_target_x` | `screen_agent.py` | Tune if targets aren't detected |

---

## Sim-to-real gap

The agent trained in sim expects clean pixel-error observations. In the real deployment:
- **Latency**: screen capture + inference + serial + muscle takes ~50–100ms
- **Non-linearity**: arm doesn't move proportionally to pulse duration
- **Fatigue**: repeated stimulation reduces response over time

**Mitigation**: after initial sim training, run 10–20 short real-arm sessions and fine-tune with `model.learn()` using real observations. The pre-trained sim weights give a huge head start.

---

## Safety checklist before EMS testing

- [ ] TENS intensity verified LOW before connecting to human
- [ ] Emergency stop button wired to cut relay power
- [ ] Max pulse duration in firmware: 200ms
- [ ] Cooldown in firmware: ≥500ms
- [ ] Stimulation site: forearm only (extensor/flexor digitorum)
- [ ] No stimulation near heart, across chest, or near head/neck
- [ ] Second person present during first EMS session
