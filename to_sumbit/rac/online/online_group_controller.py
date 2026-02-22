"""rac.online.online_group_controller

Online Group-Conditional RAC wrapper.

Same fix as global: compute s = g_hat(x, β_g) before building the set / action.
β_g calibration itself is unchanged: you pass your existing group beta routine.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class OnlineGroupRACController:
    alpha: float
    actions: Sequence[Any]
    utility_fn: Callable[..., float]

    group_beta_calibrator: Callable[..., Dict[Any, float]]
    g_hat_fn: Callable[..., float]
    set_builder: Callable[..., Any]
    decision_rule: Callable[..., Tuple[float, Any]]

    buffer_size: Optional[int] = 1000
    update_every: int = 50
    min_buffer: int = 200

    t: int = 0
    beta_by_group: Dict[Any, float] = field(default_factory=dict)
    _x_buf: List[Any] = field(default_factory=list)
    _y_buf: List[int] = field(default_factory=list)
    _g_buf: List[Any] = field(default_factory=list)

    def add_calib_point(self, x_probs: Any, y_true: Any, g: Any) -> None:
        self._x_buf.append(np.asarray(x_probs))
        self._y_buf.append(int(y_true))
        self._g_buf.append(g)
        if self.buffer_size is not None and len(self._x_buf) > self.buffer_size:
            overflow = len(self._x_buf) - self.buffer_size
            if overflow > 0:
                del self._x_buf[:overflow]
                del self._y_buf[:overflow]
                del self._g_buf[:overflow]

    def recalibrate_betas(self, **calib_kwargs) -> Dict[Any, float]:
        if len(self._x_buf) < self.min_buffer:
            return self.beta_by_group
        betas = self.group_beta_calibrator(
            self._x_buf,
            np.asarray(self._y_buf),
            np.asarray(self._g_buf),
            self.alpha,
            list(self.actions),
            self.utility_fn,
            **calib_kwargs
        )
        self.beta_by_group = {g: float(b) for g, b in betas.items()}
        return self.beta_by_group

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

    def predict_set_and_action(self, x_probs: Any, g: Any) -> Tuple[Any, int, float, float, float]:
        if g not in self.beta_by_group:
            self.recalibrate_betas()
            if g not in self.beta_by_group:
                self.beta_by_group[g] = 0.0

        beta_g = float(self.beta_by_group[g])
        s_val = float(self.g_hat_fn(x_probs, beta_g, list(self.actions), self.utility_fn))
        best_u, best_a = self.decision_rule(s_val, list(self.actions), x_probs, self.utility_fn)
        c_star = self.set_builder(x_probs, s_val, list(self.actions), self.utility_fn)

        if best_a is None:
            best_a = self._fallback_action(x_probs)
            if best_u is None:
                best_u = float("nan")

        return c_star, int(best_a), float(best_u), beta_g, s_val

    def step(self, x_probs: Any, y_true: Any, g: Any, update_buffer: bool = True, **calib_kwargs) -> Dict[str, Any]:
        self.t += 1
        y_true = int(y_true)

        c_star, a_hat, best_u, beta_g, s_val = self.predict_set_and_action(x_probs, g)

        covered = int(y_true in c_star)
        set_size = len(c_star)
        loss = 1 - covered

        if update_buffer:
            self.add_calib_point(x_probs, y_true, g)

        if (self.t % self.update_every == 0) and (len(self._x_buf) >= self.min_buffer):
            self.recalibrate_betas(**calib_kwargs)

        return {
            "t": self.t,
            "group": g,
            "beta_g": float(beta_g),
            "s_val": float(s_val),
            "y": y_true,
            "covered": covered,
            "loss": loss,
            "set_size": set_size,
            "a_hat": a_hat,
            "utility": best_u,
        }
