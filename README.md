# robot-imitation

Imitation learning for a simulated robot arm (MuJoCo `Reacher-v5`): Behavior Cloning, DAgger, and an
"Active DAgger" variant that only queries the expert when an ensemble of student policies disagrees.

**Research question:** Can a DAgger learner reach the performance of standard DAgger with fewer expert
queries if it only asks the expert in states where an ensemble of networks is uncertain?

> **Status (Nov 2026): work in progress.** Expert and evaluation are done; BC, DAgger and Active DAgger are next.

## Current results

Reference values on `Reacher-v5`, one episode per seed, seeds 0–19, mean return ± 95 % confidence
interval (t-distribution). Higher (closer to 0) is better.

| Policy | Mean return (± 95 % CI) | Episodes |
|---|---|---|
| Random actions | −43.93 ± 2.19 | 20 |
| Expert (SAC, 200k steps, training seed 0) | −3.41 ± 0.66 | 20 |

These values serve as baselines for evaluating the imitation learners. They are not hard bounds: a
learner could do worse than random or better than the expert. The expert is a single training run; it
serves as the teacher, not as a claim about SAC in general.

## Roadmap

- [x] Setup: Gymnasium + MuJoCo, CPU only
- [x] Expert: SAC (Stable-Baselines3) on `Pendulum-v1` and `Reacher-v5`
- [x] Evaluation: fixed seeds, mean return, 95 % CI (`evaluate.py`)
- [ ] Behavior Cloning: MLP in PyTorch trained on expert demonstrations, performance vs. number of demonstrations
- [ ] DAgger: student rolls out, expert labels visited states, data aggregation
- [ ] Active DAgger: ensemble disagreement decides when to query the expert, performance vs. number of queries
- [ ] Experiments: all methods over ≥ 5 training seeds with confidence intervals
- [ ] Optional: action chunking (k = 1 vs. k = 5)

## Files

| File | Purpose |
|---|---|
| `check_setup.py` | Smoke test: one random episode on Pendulum and Reacher |
| `train_expert.py` | Trains the SAC expert and saves it to `models/` |
| `watch.py` | Opens a viewer window, random actions or a trained expert |
| `evaluate.py` | Evaluates any policy (observation → action) over fixed seeds, returns mean and 95 % CI |

## Setup (CPU, no GPU needed)

```
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # first, avoids the large GPU build
pip install -r requirements.txt
python check_setup.py
```

## Reproduce

```
python train_expert.py --env Reacher-v5 --steps 200000 --seed 0   # ~45 min on a laptop CPU
python watch.py --model models/sac_Reacher-v5_seed0                 # watch the expert
python evaluate.py                                                  # random vs. expert, mean ± 95 % CI
```

Trained models are not part of the repository (`models/` is git-ignored).

The results above were produced with Python 3.12 and the pinned versions in `requirements.txt`
(Linux, CPU). Fixed seeds make runs repeatable on the same setup; other library versions or hardware
can give slightly different numbers.

## Notes

The expert uses SAC from Stable-Baselines3. Setup, expert training and viewer scripts were written with
AI assistance. I implemented the evaluation; the imitation learning methods will also be implemented by me.
