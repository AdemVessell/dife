# DIFE × Memory Vortex

**Decay-Interference Forgetting Equation + Adaptive Replay Controller**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Tests: 77 passed](https://img.shields.io/badge/tests-77%20passed-brightgreen.svg)](#test-suite)
<img src="https://img.shields.io/badge/CI-GitHub%20Actions-brightgreen" />
</p>


## What this repo is

This repository contains the **integrated DIFE × Memory Vortex controller** for continual learning.

The central idea is simple:

- **DIFE** models forgetting pressure across tasks
- **Memory Vortex** shapes replay pressure within a task
- the combined controller gives a **bounded adaptive replay policy**

This repo is the **main benchmark / integration repo**.

The standalone `memory-vortex` repository should be understood as the **component repo** for the scheduler layer. The combined claim lives here.

---

## Core claim

The current evidence in this repository supports the following:

> **The adaptive replay result belongs to the combined DIFE × Memory Vortex system, not to DIFE alone.**

On the documented split-CIFAR benchmark slice:

- `DIFE_only` saturates the replay cap and behaves like `ConstReplay_0.3`
- `Memory Vortex` provides the within-task adaptation
- `DIFE_MV` gives the best overall tradeoff among the main compared methods

---

## Main result

### Split-CIFAR-10 (5 seeds)

| Method | AA_mean | AF_mean | Replay_mean |
|:--|--:|--:|--:|
| FT | 0.7000 | 0.2713 | 0.0 |
| ConstReplay_0.1 | 0.7871 | 0.1600 | 11376.0 |
| ConstReplay_0.3 | 0.8300 | 0.1042 | 36024.0 |
| DIFE_only | 0.8300 | 0.1042 | 36024.0 |
| MV_only | 0.8313 | 0.0987 | 33511.8 |
| **DIFE_MV** | **0.8346** | **0.0962** | **32721.8** |

### Read plainly

Against `ConstReplay_0.3`, **DIFE_MV**:

- improves mean accuracy
- reduces forgetting
- uses **~9.2% less replay**

Against `MV_only`, **DIFE_MV** still improves the means on:

- accuracy
- forgetting
- replay usage

That makes the honest top-line statement:

> **The combined controller is the strongest result in the current split-CIFAR benchmark story.**

---

## Why this matters

Most replay systems are fixed:

- always 10%
- always 30%
- same policy for every task and every epoch

That is simple, but blind.

This repo explores a different control structure:

- estimate **how much forgetting pressure** the task deserves
- decide **when inside the task** replay should rise or fall
- keep the whole system **bounded**

So the question is not just:

> “Should replay happen?”

It is:

> “How much replay pressure does this task deserve, and when should it be spent?”

---

## Architecture

### DIFE — task-level forgetting model

DIFE is a closed-form forgetting equation:

```math
Q_n = \max\!\big(0,\; Q_0 \alpha^n - \beta n (1 - \alpha^n)\big)
````

It is used here as a **task-level signal**:

* fit forgetting severity from observed task history
* estimate replay pressure for the next task
* set the replay envelope / ceiling

### Memory Vortex — epoch-level scheduler

Memory Vortex is the **within-task controller**.

It reacts to a proxy forgetting signal during training and shapes replay over epochs, deciding when replay pressure should rise or fall.

### Combined controller

Conceptually:

```python
replay_fraction = clip(DIFE_envelope(task) * MV_operator(epoch), 0, r_max)
```

That gives two control scales:

* **task-level severity**
* **epoch-level timing**

---

## Honest scope

This repo does **not** claim that:

* DIFE alone is the adaptive win on split-CIFAR
* Memory Vortex is already universally dominant as a standalone scheduler
* the current results settle continual learning in general

This repo **does** claim that:

* a fitted forgetting model can provide a useful task-level replay signal
* a symbolic scheduler can shape replay within a task
* the combined controller can beat a fixed replay cap on the harder split-CIFAR setting while using less replay

---

## Benchmarks

The repository currently includes support for:

* Split-CIFAR-10
* Split-CIFAR-100
* Permuted MNIST
* MIR baseline comparisons
* capped constant replay baselines
* DIFE-only, MV-only, and combined-controller runs

Permuted MNIST remains useful, but it is the simpler setting.
The harder and more decision-relevant benchmark in this repo is **split-CIFAR**.

---

## Tests and CI

This repository includes:

* core DIFE equation tests
* broader evaluation-suite tests
* scheduler and trainer checks
* replay-budget and config coverage
* MIR and benchmark smoke coverage
* GitHub Actions CI on supported Python versions

Run locally with:

```bash
pytest tests/ -v --tb=short
```

---

## Repository layout

```text
dife.py                      # minimal standalone DIFE equation
tests/test_dife.py           # equation-level tests
tests/test_eval_suite.py     # evaluation-suite / scheduler / trainer tests
eval/                        # runners, schedulers, metrics, trainer
benchmark/                   # dataset and model factories
RESULTS_REPLICATION.md       # replication summary
PROJECT_FINAL_REPORT.md      # full report
CAVEATS.md                   # explicit limitations and corrections
```

Related component repo:

* [`memory-vortex`](https://github.com/AdemVessell/memory-vortex)

---

## Install

```bash
git clone https://github.com/AdemVessell/dife.git
cd dife
pip install -e ".[dev,bench]"
```

---

## Minimal example

```python
from dife import dife

q = dife(Q0=1.0, alpha=0.99, beta=0.001, n=100)
print(q)
```

---

## Bottom line

**DIFE × Memory Vortex is a bounded adaptive replay controller.**

In the current repo evidence:

* **DIFE alone** is not the adaptive headline on split-CIFAR
* **Memory Vortex** supplies the live replay shaping
* **the combined controller** is the result that matters

---

## Author

**Adem Vessell**

---

## License

MIT


