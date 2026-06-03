# Agent_SHOCK ⚡

**Human-robot hybrid aiming system** — an RL agent trained to control a cursor via Electrical Muscle Stimulation (EMS).

> 🧠 Built by the **Pixel Fairies**

---

## Branch Structure

| Branch | Purpose |
|---|---|
| `main` | Portfolio website (`docs/`) — hosted via GitHub Pages |
| `integration_rl_agent` | Full project code — all subsystems, run modes, integration |
| `cnn` | CNN perception pipeline — model, training, data collection, inference |

**The code lives on `integration_rl_agent`. Clone that branch to run the system.**

---

## Portfolio & Demo

- 🌐 **Project Website:** https://potatopuffs.github.io/Agent_SHOCK
- 🐙 **GitHub:** https://github.com/PotatoPuffs/Agent_SHOCK

---

## Quick Start

See the [`integration_rl_agent` branch](https://github.com/PotatoPuffs/Agent_SHOCK/tree/integration_rl_agent) and the Wiki for full setup:

- 📦 [Installation](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Installation)
- 🏃 [Running the System](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Running-the-System)

---

## Documentation (Wiki)

| Page | Description |
|---|---|
| [Project Overview](https://github.com/PotatoPuffs/Agent_SHOCK/wiki#Project-Overview) | What Agent_SHOCK is, goals, and architecture summary |
| [Subsystem Architecture](https://github.com/PotatoPuffs/Agent_SHOCK/wiki#Subsystem-Architecture) | Full system diagram and data flow |
| [Perception](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Perception) | Screen capture, CNN inference, CV pipeline |
| [Cognition](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Reinforced-Learning-Agent) | RL agent, reward structure, training |
| [Middleware](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Middleware) | micro-ROS, serial bridge, MCU comms |
| [Actuation](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Actuation-EMS) | EMS hardware, TENS relay, electrode setup |
| [Installation](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Installation) | Dependency setup and environment config |
| [Running the System](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Running-the-System) | Step-by-step run instructions |
| [Team](https://github.com/PotatoPuffs/Agent_SHOCK/wiki#Team--The-Pixel-Fairies) | The Pixel Fairies |

---

## Safety

⚠️ This system delivers electrical stimulation to a human. See the [EMS Safety Checklist](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Actuation-EMS#safety-checklist) before any hardware testing.

---

