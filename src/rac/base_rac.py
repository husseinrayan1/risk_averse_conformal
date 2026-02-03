from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence
import math
import numpy as np

ScoreFn = Callable[[np.ndarray, Any], float]


@dataclass(frozen=True)
class RACResult:
    alpha: float
    beta: float
    n_cal: int


class BaseRAC:
    """
    Working offline split-conformal prediction sets:
        C(x; beta) = { y in Y : score(x,y) <= beta }

    beta is calibrated from (X_cal, Y_cal) to guarantee
    marginal coverage >= 1 - alpha (under exchangeability).
    """

    def __init__(self, alpha: float, y_space: Sequence[Any], score_fn: ScoreFn):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        if len(y_space) == 0:
            raise ValueError("y_space must be non-empty.")
        self.alpha = float(alpha)
        self.y_space = list(y_space)
        self.score_fn = score_fn

        self._is_calibrated = False
        self.result: Optional[RACResult] = None

    @staticmethod
    def _conformal_beta(scores: np.ndarray, alpha: float) -> float:
        n = len(scores)
        k = int(math.ceil((n + 1) * (1.0 - alpha)))  # 1-indexed rank
        k = min(max(k, 1), n)
        return float(np.partition(scores, k - 1)[k - 1])

    def fit(self, X_cal: np.ndarray, Y_cal: np.ndarray) -> RACResult:
        X_cal = np.asarray(X_cal)
        Y_cal = np.asarray(Y_cal, dtype=object)
        if len(X_cal) != len(Y_cal):
            raise ValueError("X_cal and Y_cal must have the same length.")
        n = len(Y_cal)

        scores = np.empty(n, dtype=float)
        for i in range(n):
            scores[i] = float(self.score_fn(X_cal[i], Y_cal[i]))

        beta = self._conformal_beta(scores, self.alpha)

        self.result = RACResult(alpha=self.alpha, beta=beta, n_cal=n)
        self._is_calibrated = True
        return self.result

    def predict_set(self, x: np.ndarray) -> List[Any]:
        if not self._is_calibrated or self.result is None:
            raise RuntimeError("Call fit() before predict_set().")

        x = np.asarray(x)
        beta = self.result.beta
        return [y for y in self.y_space if float(self.score_fn(x, y)) <= beta]
