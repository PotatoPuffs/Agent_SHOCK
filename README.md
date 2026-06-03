# Agent_SHOCK ⚡ — `integration_rl_agent`
 
This branch contains the full project code for Agent_SHOCK — all subsystems, run modes, and integration.
 
> For the portfolio website, see the [`main` branch](https://github.com/PotatoPuffs/Agent_SHOCK/tree/main).
> For full documentation, see the [Wiki](https://github.com/PotatoPuffs/Agent_SHOCK/wiki).
 
---
 
## Installation
 
### 1 — Clone this branch
 
```bash
git clone -b integration_rl_agent https://github.com/PotatoPuffs/Agent_SHOCK.git
cd Agent_SHOCK
```
 
### 2 — Install dependencies
 
```bash
pip install -r requirements.txt
```
 
---
 
## Run Modes
 
| Mode | What it does | Hardware needed |
|---|---|---|
| `train` | Trains the PPO agent in simulation | None |
| `eval` | Evaluates the saved model | None |
| `test_deploy` | Tests EMS in isolation from perception | EMS optional |
| `deploy` | Live loop — observer → agent → EMS | Depends on flags |
| `hsv_test` | Verifies HSV detection before a live run | None |
 
### Quick start — fully simulated (no hardware required)
```bash
python run.py --mode deploy --cnn sim --ems sim
```
 
### Live closed loop on a person
```bash
python run.py --mode deploy --cnn hsv --ems real
```
 
### Train the RL agent
```bash
python run.py --mode train --steps 250000
```
 
See the [Running the System](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Running-the-System) wiki page for all CLI options and flag combinations.
 
---
 
## Full Documentation
 
- 📦 [Installation](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Installation)
- 🏃 [Running the System](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Running-the-System)
- 🏗️ [Subsystem Architecture](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Subsystem-Architecture)
- 🌐 [Portfolio](https://potatopuffs.github.io/Agent_SHOCK)
---
 
## Safety
 
⚠️ This system delivers electrical stimulation to a human. See the [EMS Safety Checklist](https://github.com/PotatoPuffs/Agent_SHOCK/wiki/Actuation-The-Muscles#safety-checklist) before any hardware testing.
