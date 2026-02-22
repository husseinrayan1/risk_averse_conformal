"""experiments.online_plots

Plotting utilities for online experiments:
- rolling miscoverage (loss)
- beta trajectory
- rolling utility / set size
"""

from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt


def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if win <= 1:
        return x
    out = np.full_like(x, np.nan, dtype=float)
    c = np.cumsum(np.nan_to_num(x))
    for i in range(len(x)):
        j0 = max(0, i - win + 1)
        denom = (i - j0 + 1)
        out[i] = (c[i] - (c[j0 - 1] if j0 > 0 else 0.0)) / denom
    return out


def plot_online_global(rows: List[Dict[str, Any]], title: str = "", roll: int = 200, shift_at: int | None = None):
    t = np.array([r["t"] for r in rows])
    loss = np.array([r["loss"] for r in rows], dtype=float)
    beta = np.array([r["beta"] for r in rows], dtype=float)
    util = np.array([r.get("utility", np.nan) for r in rows], dtype=float)
    set_size = np.array([r.get("set_size", np.nan) for r in rows], dtype=float)

    # plt.figure()
    # plt.plot(t, _rolling_mean(loss, roll))
    # plt.axhline(np.nanmean(loss), linestyle="--")
    # plt.xlabel("t"); plt.ylabel(f"Rolling miscoverage (win={roll})")
    # plt.title(title + " - Rolling miscoverage")
    # plt.show()
    plt.figure()
    plt.plot(t, _rolling_mean(loss, roll), label="Online")
    plt.axhline(np.nanmean(loss), linestyle="--", label="Mean risk")

    if shift_at is not None:
        plt.axvline(shift_at, color='red', linestyle='--', label='Shift')

    plt.xlabel("t")
    plt.ylabel(f"Rolling miscoverage (win={roll})")
    plt.title(title + " - Rolling miscoverage")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t, beta)
    plt.xlabel("t"); plt.ylabel("beta(t)")
    plt.title(title + " - beta trajectory")
    plt.show()

    plt.figure()
    plt.plot(t, _rolling_mean(util, roll))
    plt.xlabel("t"); plt.ylabel(f"Rolling utility (win={roll})")
    plt.title(title + " - Rolling utility")
    plt.show()

    plt.figure()
    plt.plot(t, _rolling_mean(set_size, roll))
    plt.xlabel("t"); plt.ylabel(f"Rolling set size (win={roll})")
    plt.title(title + " - Rolling set size")
    plt.show()


def plot_online_group(rows: List[Dict[str, Any]], title: str = "", roll: int = 200):
    t = np.array([r["t"] for r in rows])
    groups = [r["group"] for r in rows]
    uniq = sorted(list(set(groups)), key=lambda x: str(x))

    plt.figure()
    for g in uniq:
        idx = np.array([gg == g for gg in groups])
        loss = np.array([rows[i]["loss"] for i in range(len(rows))], dtype=float)
        win = max(10, min(roll, max(10, int(idx.sum() / 5))))
        plt.plot(t[idx], _rolling_mean(loss[idx], win), label=f"g={g}")
    plt.xlabel("t"); plt.ylabel("Rolling miscoverage")
    plt.title(title + " - Rolling miscoverage per group")
    plt.legend(); plt.show()

    plt.figure()
    for g in uniq:
        idx = np.array([gg == g for gg in groups])
        beta_g = np.array([rows[i]["beta_g"] for i in range(len(rows))], dtype=float)
        plt.plot(t[idx], beta_g[idx], label=f"g={g}")
    plt.xlabel("t"); plt.ylabel("beta_g(t)")
    plt.title(title + " - beta per group")
    plt.legend(); plt.show()
