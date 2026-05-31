# Agent_SHOCK ⚡

**Human-robot hybrid aiming system** — an Reinforced Learning (RL) agent trained to control a cursor via Electrical Muscle Stimulation (EMS).

> 🧠 Built by the **Pixel Fairies** for [AI in Robotics]

---

## Project Context

Agent_SHOCK is a closed-loop computer interface system where a reinforcement learning agent learns to aim a cursor at targets in a browser-based game [aiming.pro](https://aiming.pro/app#/training/drills/52502) → then actuates a real human arm using a TENS/EMS device to physically move the mouse.

The system is split into four subsystems: **Perception**, **Cognition**, **Middleware**, and **Actuation** — detailed documentation for each lives in the [Wiki](https://github.com/PotatoPuffs/Agent_SHOCK/wiki).

---

## Repository Structure

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

## Quick Start

See the Wiki for full setup and usage guides:

- 📦 [Installation Guide](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Installation)
- 🏃 [Running the System](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Running-the-System)
- ⚡ [EMS / Hardware Setup](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Actuation-The-Muscles)

---

## Documentation (Wiki)

Full project documentation is maintained in the [Wiki](https://github.com/PotatoPuffs/Agent_SHOCK/wiki):

| Page | Description |
|---|---|
| [Project Overview](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Project-Overview) | What Agent_SHOCK is, goals, and architecture summary |
| [Subsystem Architecture](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Subsystem-Architecture) | Full system diagram and data flow |
| [Perception](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Perception-The-Eyes) | Screen capture, CNN inference, CV pipeline |
| [Cognition](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Cognition-The-Brain) | RL agent, reward structure, training |
| [Middleware](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Middleware-The-Nervous-System) | micro-ROS, serial bridge, MCU comms |
| [Actuation](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Actuation-The-Muscles) | EMS hardware, TENS relay, electrode setup |
| [Installation](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Installation) | Dependency setup and environment config |
| [Running the System](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Running-the-System) | Step-by-step run instructions |
| [Team](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Team) | The Pixel Fairies |

---

## Portfolio & Demo

- 🌐 **Project Website:** [Coming soon — link to be added]
- 🎥 **Demo Video:** [Coming soon — link to be added]

---

## Safety

⚠️ This system delivers electrical stimulation to a human. See the [EMS Safety Checklist](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Actuation-The-Muscles#safety-checklist) before any hardware testing.

---

## Team

Built by the **Pixel Fairies** — see the [Team Wiki page](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Team) or our [project portfolio](<!-- link -->).
