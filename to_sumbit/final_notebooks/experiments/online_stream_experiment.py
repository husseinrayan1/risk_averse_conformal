"""experiments.online_stream_experiment

Streaming experiment utilities for Online RAC / Online Group-RAC.

- StreamScenario: stationarity / shift specification
- make_stream: yields (x, y, g) tuples; can apply synthetic drift after shift
- run_online_global / run_online_group: run a controller on a stream and collect logs
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class StreamScenario:
    name: str = "stationary"
    shift_at: Optional[int] = None
    post_shift_group_probs: Optional[Dict[Any, float]] = None
    corrupt_groups: Optional[List[Any]] = None
    corrupt_strength: float = 0.3


def _corrupt_probs(p: np.ndarray, strength: float) -> np.ndarray:
    p = np.asarray(p, dtype=float).reshape(-1)
    u = np.ones_like(p) / max(1, p.size)
    out = (1 - strength) * p + strength * u
    out = np.clip(out, 1e-12, None)
    out = out / out.sum()
    return out


def make_stream(
    probs: Sequence[Any],
    labels: Sequence[int],
    groups: Optional[Sequence[Any]],
    scenario: StreamScenario,
    rng_seed: int = 0,
) -> Iterator[Tuple[Any, int, Optional[Any]]]:
    rng = np.random.default_rng(rng_seed)
    T = len(labels)

    probs = list(probs)
    labels = list(labels)

    if groups is None:
        groups = [None] * T
    else:
        groups = list(groups)

    for t in range(T):
        g = groups[t]

        if scenario.shift_at is not None and t >= scenario.shift_at and scenario.post_shift_group_probs is not None:
            keys = list(scenario.post_shift_group_probs.keys())
            w = np.array([scenario.post_shift_group_probs[k] for k in keys], dtype=float)
            w = w / w.sum()
            g = rng.choice(keys, p=w)

        x = probs[t]
        y = int(labels[t])

        if scenario.shift_at is not None and t >= scenario.shift_at:
            if scenario.corrupt_groups is None:
                x = _corrupt_probs(np.asarray(x), scenario.corrupt_strength)
            else:
                if g in scenario.corrupt_groups:
                    x = _corrupt_probs(np.asarray(x), scenario.corrupt_strength)

        yield (x, y, g)


def run_online_global(stream: Iterable[Tuple[Any, int, Any]], controller):
    rows = []
    for (x, y, _g) in stream:
        rows.append(controller.step(x, y, update_buffer=True))
    return rows


def run_online_group(stream: Iterable[Tuple[Any, int, Any]], controller):
    rows = []
    for (x, y, g) in stream:
        rows.append(controller.step(x, y, g, update_buffer=True))
    return rows
