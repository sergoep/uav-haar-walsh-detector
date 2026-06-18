# UAV Haar--Walsh Synthetic and Real-Data Validation Code

This repository contains reproducible Python code associated with the paper:

**Calibrated Haar--Walsh Structural Features for Lightweight Interference Detection in Short UAV Signal Windows**

The repository provides two reproducibility components:

1. A synthetic benchmark protocol for short UAV-related signal windows.
2. A real-data validation runner for public GNSS/RF I/Q datasets.

The code implements Haar--Walsh structural feature extraction, baseline score computation, robust median/MAD normalization, split-conformal thresholding, ROC/AUC evaluation, conformal FPR/TPR checks, timing measurements, and LaTeX/figure output generation.

## Contents

- `run_synthetic_uav_haar_walsh.py` — synthetic benchmark protocol for short UAV-related signal windows.
- `run_real_gnss_haar_walsh.py` — real-data validation runner for public GNSS/RF I/Q datasets.
- `requirements.txt` — Python dependencies.
- `CITATION.cff` — citation metadata.
- `LICENSE` — MIT license.
- `README.md` — repository documentation.

## Reproducibility

Python version: 3.10 or later.

Install dependencies:

```bash
pip install -r requirements.txt
