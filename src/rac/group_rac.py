from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence
import numpy as np

from .base_rac import BaseRAC, ScoreFn


@dataclass(frozen=True)
class GroupRACResult:
    alpha: float
    groups: List[Any]
    n_per_group: Dict[Any, int]


class GroupConditionalRAC:
    """
    Group-Conditional split-conformal prediction sets.

    - Partition calibration data by group g = group_fn(x)
    - Fit one BaseRAC per group
    - At test time, use the calibrator of the test point's group
    """

    def __init__(
        self,
        alpha: float,
        group_fn: Callable[[np.ndarray], Any],
        y_space: Sequence[Any],
        score_fn: ScoreFn,
    ):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = float(alpha)
        self.group_fn = group_fn
        self.y_space = list(y_space)
        self.score_fn = score_fn

        self.models: Dict[Any, BaseRAC] = {}
        self.result: Optional[GroupRACResult] = None
        self._is_calibrated = False

    def fit(self, X_cal: np.ndarray, Y_cal: np.ndarray) -> GroupRACResult:
        X_cal = np.asarray(X_cal)
        Y_cal = np.asarray(Y_cal, dtype=object)
        if len(X_cal) != len(Y_cal):
            raise ValueError("X_cal and Y_cal must have the same length.")

        groups = np.array([self.group_fn(x) for x in X_cal], dtype=object)
        unique_groups = list(np.unique(groups))

        n_per_group: Dict[Any, int] = {}
        self.models = {}

        for g in unique_groups:
            idx = np.where(groups == g)[0]
            n_per_group[g] = int(len(idx))

            if len(idx) == 0:
                continue

            m = BaseRAC(alpha=self.alpha, y_space=self.y_space, score_fn=self.score_fn)
            m.fit(X_cal[idx], Y_cal[idx])
            self.models[g] = m

        self.result = GroupRACResult(alpha=self.alpha, groups=unique_groups, n_per_group=n_per_group)
        self._is_calibrated = True
        return self.result

    def predict_set(self, x: np.ndarray) -> List[Any]:
        if not self._is_calibrated:
            raise RuntimeError("Call fit() before predict_set().")

        x = np.asarray(x)
        g = self.group_fn(x)

        if g not in self.models:
            raise KeyError(
                f"Group {g!r} not seen during calibration. "
                f"Seen groups: {list(self.models.keys())}"
            )

        return self.models[g].predict_set(x)
