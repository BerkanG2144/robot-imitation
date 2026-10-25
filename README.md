# robot-imitation

Imitation Learning in einer Robotik-Simulation: Behavior Cloning, DAgger und eine Variante, die den Experten nur bei Unsicherheit eines Ensembles abfragt.

_Status: im Aufbau._

## Setup (CPU, keine GPU nötig)
```
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # zuerst, sonst wird eine große GPU-Version geladen
pip install -r requirements.txt
python check_setup.py
```

## Experte trainieren
```
python train_expert.py --env Pendulum-v1 --steps 10000
```
