"""rac.online.online_controller

Online wrapper around your notebook RAC pipeline.

CRITICAL: In your offline code, β is NOT passed directly into set construction.
You must compute s = g_hat(x, β) first:
    s = compute_g_hat(x_probs, beta, actions, utility_fn)
    C = get_conformal_set(x_probs, s, actions, utility_fn)
    (u, a) = hbtheta_and_arg(s, actions, x_probs, utility_fn)

This module keeps β calibration unchanged: you still pass your existing `find_threshold_q`.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class OnlineRACController:
    alpha: float
    actions: Sequence[Any]
    utility_fn: Callable[..., float]

    beta_calibrator: Callable[..., float]
    g_hat_fn: Callable[..., float]
    set_builder: Callable[..., Any]
    decision_rule: Callable[..., Tuple[float, Any]]

    buffer_size: Optional[int] = 1000
    update_every: int = 50
    min_buffer: int = 200

    beta: Optional[float] = None
    t: int = 0
    _x_buf: List[Any] = field(default_factory=list)
    _y_buf: List[int] = field(default_factory=list)

    def add_calib_point(self, x_probs: Any, y_true: Any) -> None:
        self._x_buf.append(np.asarray(x_probs))
        self._y_buf.append(int(y_true))
        if self.buffer_size is not None and len(self._x_buf) > self.buffer_size:
            overflow = len(self._x_buf) - self.buffer_size
            if overflow > 0:
                del self._x_buf[:overflow]
                del self._y_buf[:overflow]

    def recalibrate_beta(self, **calib_kwargs) -> Optional[float]:
        if len(self._x_buf) < self.min_buffer:
            return self.beta
        self.beta = float(self.beta_calibrator(
            self._x_buf,                 # list-of-arrays (matches notebook)
            np.asarray(self._y_buf),
            self.alpha,
            list(self.actions),
            self.utility_fn,
            **calib_kwargs
        ))
        return self.beta

    def _fallback_action(self, x_probs: Any) -> Any:
        acts = list(self.actions)
        x_arr = np.asarray(x_probs)
        try:
            idx = int(np.argmax(x_arr))
            if 0 <= idx < len(acts):
                return acts[idx]
        except Exception:
            pass
        return acts[0] if acts else 0

    def predict_set_and_action(self, x_probs: Any) -> Tuple[Any, int, float, float]:
        if self.beta is None:
            self.recalibrate_beta()
            if self.beta is None:
                self.beta = 0.0

        beta = float(self.beta)
        s_val = float(self.g_hat_fn(x_probs, beta, list(self.actions), self.utility_fn))
        best_u, best_a = self.decision_rule(s_val, list(self.actions), x_probs, self.utility_fn)
        c_star = self.set_builder(x_probs, s_val, list(self.actions), self.utility_fn)

        if best_a is None:
            best_a = self._fallback_action(x_probs)
            if best_u is None:
                best_u = float("nan")

        return c_star, int(best_a), float(best_u), s_val

    def step(self, x_probs: Any, y_true: Any, update_buffer: bool = True, **calib_kwargs) -> Dict[str, Any]:
        self.t += 1
        c_star, a_hat, best_u, s_val = self.predict_set_and_action(x_probs)
        y_true = int(y_true)

        covered = int(y_true in c_star)
        set_size = len(c_star)
        loss = 1 - covered

        if update_buffer:
            self.add_calib_point(x_probs, y_true)

        if (self.t % self.update_every == 0) and (len(self._x_buf) >= self.min_buffer):
            self.recalibrate_beta(**calib_kwargs)

        return {
            "t": self.t,
            "beta": float(self.beta) if self.beta is not None else np.nan,
            "s_val": float(s_val),
            "y": y_true,
            "covered": covered,
            "loss": loss,
            "set_size": set_size,
            "a_hat": a_hat,
            "utility": best_u,
        }
