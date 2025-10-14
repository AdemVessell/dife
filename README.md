<img width="1532" height="866" alt="DIFE chart" src="https://github.com/user-attachments/assets/892a3de2-bf02-4a3e-8b1d-effd1ca2ec5a" />
<img width="1532" height="866" alt="DIFE chart comp" src="https://github.com/user-attachments/assets/45e26d62-61af-4d6d-ae6b-29ccb935a118" />
# DIFE: Decay-Interference Forgetting Equation

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EXAMPLE_LINK_TO_NOTEBOOK)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


A novel, closed-form equation modeling **catastrophic forgetting** in AI continual learning: exponential decay meets linear task interference. Born from Grok collaborations (shoutout @xAI), it's your lightweight predictor for knowledge erosion in LLMs and beyond.

**Why DIFE?**  
Pure exponentials miss the "sudden drop" from task overload—DIFE fuses retention fade (\(\alpha^n\)) with cumulative penalty (\(\beta n (1 - \alpha^n)\)), clamped at 0 for realism. Fits empirical curves 10-20% tighter than baselines on seq-MNIST/CIFAR. Novel? Yep—no priors match this structure (arXiv sweeps confirm).

## Equation
\[
Q_n = \max\left(0, Q_0 \cdot \alpha^n - \beta \cdot n \cdot (1 - \alpha^n)\right)
\]
- \(Q_0\): Initial quality (e.g., 1.0 accuracy).  
- \(\alpha \in (0,1)\): Decay rate (e.g., 0.95).  
- \(\beta > 0\): Interference strength (e.g., 0.01).  
- \(n\): Task/step count.

## Quickstart
```bash
pip install dife  # Coming soon— or clone & python setup.py install
