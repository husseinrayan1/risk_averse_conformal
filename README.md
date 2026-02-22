# Risk-Averse Conformal (RAC) — Reliability in ML Final Project

GitHub (this repo): https://github.com/husseinrayan1/risk_averse_conformal

This project implements and evaluates **Risk-Averse Calibration (RAC)** for decision-aware conformal prediction, and extends it with:
1) **Group-Conditional RAC** (gender groups),
2) **Adaptive groups** via clustering (calibration-time),
3) **Online / Streaming RAC** with periodic recalibration under **stationary** and **drift** settings.

> **Main reproduction entry point:** run the notebooks inside `to_sumbit/` (see below).

---

## Repository structure (what matters for the submission)

- `to_sumbit/`  
  The submission folder containing:
  - final notebooks to reproduce all experiments
  - generated plots used in the report
  - online streaming code (controller + streaming experiment)

- `src/`, `rac/`, `experiments/`, `utils/`  
  Supporting code used by the notebooks (some copies also exist under `to_sumbit/`).

---

## Environment setup

### Option A — Conda (recommended if you already use it)
```bash
conda create -n rac_env python=3.11 -y
conda activate rac_env
pip install -r requirements.txt
