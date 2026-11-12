# robot-imitation

Imitation learning for a simulated robot arm (MuJoCo `Reacher-v5`): Behavior Cloning, DAgger, and an
"Active DAgger" variant that only queries the expert when an ensemble of student policies disagrees.

**Research question:** Can a DAgger learner reach the performance of standard DAgger with fewer expert
queries if it only asks the expert in states where an ensemble of networks is uncertain?

> **Status (Nov 2026): work in progress.** Expert, evaluation and a first Behavior Cloning baseline are done;
> DAgger and Active DAgger are next.

## Current results

Reacher-v5, one episode per seed, evaluation seeds 0–19, mean return ± 95 % confidence interval
(t-distribution). Higher (closer to 0) is better.

| Policy | Mean return (± 95 % CI) | Episodes |
|---|---|---|
| Random actions | −43.93 ± 2.19 | 20 |
| Expert (SAC, 200k steps, training seed 0) | −3.41 ± 0.66 | 20 |

### Behavior Cloning (preliminary)

MLP 10→64→64→2 (ReLU, `tanh` output), Adam, 300 full-batch epochs, no input normalization. Demonstrations
come from the expert on seeds ≥ 1000, evaluation on seeds 0–19. **One training run per row**, so the
intervals only reflect variation across test episodes, not across training runs.

| Demo episodes | Samples | Mean return (± 95 % CI) |
|---|---|---|
| 1 | 50 | −9.76 ± 1.87 |
| 5 | 250 | −25.37 ± 10.99 |
| 10 | 500 | −10.42 ± 3.52 |
| 20 | 1000 | −8.88 ± 3.85 |
| 50 | 2500 | −7.52 ± 3.74 |

All BC variants beat random actions but stay below the expert. The 5-episode result is worse than the
1-episode result, which I cannot explain with a single run per row. Multiple training seeds, input
normalization and longer training are the next checks.

## Files

| File | Purpose |
|---|---|
| `check_setup.py` | Smoke test: one random episode on Pendulum and Reacher |
| `train_expert.py` | Trains the SAC expert and saves it to `models/` |
| `watch.py` | Opens a viewer window, random actions or a trained expert |
| `evaluate.py` | Evaluates any policy (observation → action) over fixed seeds, returns mean and 95 % CI |
| `train_bc.py` | Collects expert demonstrations, trains the BC network, sweeps the number of demonstrations |

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
python train_bc.py                                                  # BC sweep over demonstration counts
```

Trained models are not part of the repository (`models/` is git-ignored).

The results above were produced with Python 3.12 and the pinned versions in `requirements.txt`
(Linux, CPU). Fixed seeds make runs repeatable on the same setup; other library versions or hardware
can give slightly different numbers.

## Notes

Setup, expert training, the viewer script and the first version of the BC code were written with AI