# src/rac/paper_rac.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _upper_quantile_value(values: np.ndarray, probs: np.ndarray, t: float) -> float:
    """
    Discrete 'upper-quantile' used for theta(x,t):
    q_a(t) := sup { q : P( U_a >= q ) >= t }.

    Implementation: sort utilities descending and accumulate probability mass.
    Return the utility value at which cumulative mass first reaches >= t.
    """
    # values: shape (K,), probs: shape (K,)
    order = np.argsort(-values)
    v_sorted = values[order]
    p_sorted = probs[order]
    cum = 0.0
    for v, p in zip(v_sorted, p_sorted):
        cum += float(p)
        if cum + 1e-12 >= t:  # small tolerance
            return float(v)
    # if t is extremely close to 1 and numeric issues: return min utility
    return float(v_sorted[-1])


def _candidate_t_values_for_action(values: np.ndarray, probs: np.ndarray) -> List[float]:
    """
    Returns breakpoints of t where q_a(t) can change:
    cumulative masses when sorting utilities in descending order.
    """
    order = np.argsort(-values)
    p_sorted = probs[order]
    cums = np.cumsum(p_sorted)
    # include 0 as candidate (edge)
    out = [0.0]
    for c in cums:
        out.append(float(c))
    # clamp to [0,1]
    out = [min(1.0, max(0.0, t)) for t in out]
    # unique-ish
    out = sorted(set(out))
    return out


@dataclass
class _ThetaAction:
    theta: float
    action: int


class PaperRAC:
    """
    Paper-aligned Risk-Averse Calibration (RAC) for discrete labels and discrete actions.

    Assumptions for X:
      - X is an array-like of shape (n, K + ...), where X[i][:K] are estimated class probabilities p(y|x).
      - y in {0,...,K-1}

    Parameters
    ----------
    alpha : float
        Miscoverage level.
    U : np.ndarray
        Utility matrix of shape (A, K) where U[a, y] = utility for action a when true label is y.
    y_space : Sequence[int]
        Labels, typically range(K).
    beta_bounds : Tuple[float,float]
        Search bounds for beta in the conformal feasibility problem.
    beta_tol : float
        Binary search tolerance for beta.
    max_iter : int
        Max iterations for each beta binary search.
    """

    def __init__(
        self,
        alpha: float,
        U: np.ndarray,
        y_space: Sequence[int],
        beta_bounds: Tuple[float, float] = (-20.0, 20.0),
        beta_tol: float = 1e-3,
        max_iter: int = 50,
    ):
        self.alpha = float(alpha)
        self.U = np.asarray(U, dtype=float)
        self.y_space = list(map(int, y_space))
        self.K = len(self.y_space)
        self.A = int(self.U.shape[0])

        assert self.U.shape == (self.A, self.K), f"U must be (A,K) but got {self.U.shape}"

        self.beta_lo, self.beta_hi = float(beta_bounds[0]), float(beta_bounds[1])
        self.beta_tol = float(beta_tol)
        self.max_iter = int(max_iter)

        # calibration storage
        self._P_cal: Optional[np.ndarray] = None  # shape (n, K)
        self._y_cal: Optional[np.ndarray] = None  # shape (n,)

    # ---------------------------
    # Core paper functions: theta, a, g
    # ---------------------------
    def _theta_and_action(self, p: np.ndarray, t: float) -> _ThetaAction:
        """
        theta(x,t) = max_a q_a(t), with q_a(t) defined from utilities under p.
        Also returns an argmax action a(x,t).
        """
        best_theta = None
        best_a = 0
        for a in range(self.A):
            u_vals = self.U[a, :]  # utilities for all labels
            q = _upper_quantile_value(u_vals, p, t)
            if best_theta is None or q > best_theta:
                best_theta = q
                best_a = a
        return _ThetaAction(theta=float(best_theta), action=int(best_a))

    def _g(self, p: np.ndarray, beta: float) -> Tuple[float, float, int]:
        """
        g(x,beta) = argmax_{t in [0,1]} theta(x,t) + beta*t.
        Returns (t*, theta(x,t*), a(x,t*)).
        Deterministic tie-break: prefer larger t on ties (more conservative).
        """
        eps = 1e-12

        cand_t = set([0.0, 1.0])
        for a in range(self.A):
            cand_t.update(_candidate_t_values_for_action(self.U[a, :], p))
        cand_t = sorted(cand_t)

        best_obj = -np.inf
        best_t = 0.0
        best_theta = None
        best_a = 0

        for t in cand_t:
            ta = self._theta_and_action(p, t)
            obj = float(ta.theta) + float(beta) * float(t)

            # tie-break: prefer larger t
            if (obj > best_obj + eps) or (abs(obj - best_obj) <= eps and t > best_t):
                best_obj = obj
                best_t = float(t)
                best_theta = float(ta.theta)
                best_a = int(ta.action)

        assert best_theta is not None
        return best_t, best_theta, best_a

    def _C_hat(self, p: np.ndarray, beta: float) -> List[int]:
        """
        \hat C(x;beta) = { y : u(\hat a(x,beta), y) >= \hat theta(x,beta) }.
        Safety: never return empty set.
        """
        _, theta_hat, a_hat = self._g(p, beta)
        u_row = self.U[a_hat, :]

        C = [y for y in self.y_space if float(u_row[y]) + 1e-12 >= float(theta_hat)]
        if len(C) == 0:
            C = [int(np.argmax(u_row))]  # guarantee non-empty
        return C


    # ---------------------------
    # Conformal feasibility problem for each candidate y
    # ---------------------------
    def _feasible(self, beta: float, y_candidate: int, p_test: np.ndarray) -> bool:
        """
        Check the paper constraint:
          (sum_i 1[y_i in C_hat(p_i;beta)] + 1[y_candidate in C_hat(p_test;beta)]) / (n+1) >= 1-alpha
        """
        assert self._P_cal is not None and self._y_cal is not None
        P = self._P_cal
        y_cal = self._y_cal
        n = len(y_cal)

        # compute indicator over calibration points
        count = 0
        for i in range(n):
            C_i = self._C_hat(P[i], beta)
            if int(y_cal[i]) in C_i:
                count += 1

        # add hallucinated test label
        C_test = self._C_hat(p_test, beta)
        if int(y_candidate) in C_test:
            count += 1

        return (count / float(n + 1)) + 1e-12 >= (1.0 - self.alpha)

    def _solve_beta_for_y(self, y_candidate: int, p_test: np.ndarray) -> float:
        """
        Solve beta_y = min beta s.t. feasibility holds.
        Uses binary search on [lo, hi], expanding hi if needed, and pushing lo left if lo is feasible.
        """
        lo = float(self.beta_lo)
        hi = float(self.beta_hi)

        # ensure feasible at hi by expanding if necessary
        it_expand = 0
        while not self._feasible(hi, y_candidate, p_test) and it_expand < 20:
            hi *= 2.0
            it_expand += 1

        if not self._feasible(hi, y_candidate, p_test):
            return hi  # best effort if bounds too small

        # make lo likely infeasible by shifting left in a controlled way
        it_shrink = 0
        while self._feasible(lo, y_candidate, p_test) and it_shrink < 20:
            lo -= (hi - lo) + 1.0
            it_shrink += 1

        # binary search
        for _ in range(self.max_iter):
            mid = 0.5 * (lo + hi)
            if self._feasible(mid, y_candidate, p_test):
                hi = mid
            else:
                lo = mid
            if abs(hi - lo) <= self.beta_tol:
                break

        return float(hi)


    # ---------------------------
    # Public API
    # ---------------------------
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        X_cal = np.asarray(X_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=int).reshape(-1)
        assert X_cal.shape[0] == y_cal.shape[0]
        assert X_cal.shape[1] >= self.K

        self._P_cal = X_cal[:, : self.K].copy()
        self._y_cal = y_cal.copy()
        return self

    def predict_set(self, x: np.ndarray) -> List[int]:
        """
        Algorithm 1 output:
          C_RAC(x) = { y : y in C_hat(x; beta_y) }
        """
        assert self._P_cal is not None, "Call fit() first."
        x = np.asarray(x, dtype=float)
        p_test = x[: self.K].astype(float)

        out = []
        for y in self.y_space:
            beta_y = self._solve_beta_for_y(int(y), p_test)
            C_test = self._C_hat(p_test, beta_y)
            if int(y) in C_test:
                out.append(int(y))
        return out


class GroupPaperRAC(PaperRAC):
    """
    Group-conditional version:
    Solve the Algorithm 1 feasibility constraint inside the test point's group only.

    group_fn(x) -> int group id.
    """

    def __init__(
        self,
        alpha: float,
        U: np.ndarray,
        y_space: Sequence[int],
        group_fn: Callable[[np.ndarray], int],
        beta_bounds: Tuple[float, float] = (-20.0, 20.0),
        beta_tol: float = 1e-3,
        max_iter: int = 50,
    ):
        super().__init__(alpha, U, y_space, beta_bounds, beta_tol, max_iter)
        self.group_fn = group_fn
        self._groups: Optional[np.ndarray] = None  # calibration groups
        self._idx_by_group: Dict[int, np.ndarray] = {}

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        super().fit(X_cal, y_cal)
        X_cal = np.asarray(X_cal, dtype=float)
        groups = np.array([int(self.group_fn(X_cal[i])) for i in range(X_cal.shape[0])], dtype=int)
        self._groups = groups
        self._idx_by_group = {}
        for g in np.unique(groups):
            self._idx_by_group[int(g)] = np.where(groups == g)[0]
        return self

    def _feasible_group(self, beta: float, y_candidate: int, p_test: np.ndarray, g_test: int) -> bool:
        """
        Group-only constraint:
          (sum_{i in G} 1[y_i in C_hat(p_i;beta)] + 1[y_candidate in C_hat(p_test;beta)]) / (n_g+1) >= 1-alpha
        """
        assert self._P_cal is not None and self._y_cal is not None and self._groups is not None
        idx = self._idx_by_group.get(int(g_test), None)
        if idx is None or len(idx) == 0:
            # fallback to global
            return self._feasible(beta, y_candidate, p_test)

        P = self._P_cal
        y_cal = self._y_cal
        n_g = len(idx)

        count = 0
        for i in idx:
            C_i = self._C_hat(P[i], beta)
            if int(y_cal[i]) in C_i:
                count += 1

        C_test = self._C_hat(p_test, beta)
        if int(y_candidate) in C_test:
            count += 1

        return (count / float(n_g + 1)) + 1e-12 >= (1.0 - self.alpha)

    def _solve_beta_for_y_group(self, y_candidate: int, p_test: np.ndarray, g_test: int) -> float:
        lo = float(self.beta_lo)
        hi = float(self.beta_hi)

        it_expand = 0
        while not self._feasible_group(hi, y_candidate, p_test, g_test) and it_expand < 20:
            hi *= 2.0
            it_expand += 1
        if not self._feasible_group(hi, y_candidate, p_test, g_test):
            return hi

        it_shrink = 0
        while self._feasible_group(lo, y_candidate, p_test, g_test) and it_shrink < 20:
            lo -= (hi - lo) + 1.0
            it_shrink += 1

        for _ in range(self.max_iter):
            mid = 0.5 * (lo + hi)
            if self._feasible_group(mid, y_candidate, p_test, g_test):
                hi = mid
            else:
                lo = mid
            if abs(hi - lo) <= self.beta_tol:
                break

        return float(hi)


    def predict_set(self, x: np.ndarray) -> List[int]:
        assert self._P_cal is not None, "Call fit() first."
        x = np.asarray(x, dtype=float)
        p_test = x[: self.K].astype(float)
        g_test = int(self.group_fn(x))

        out = []
        for y in self.y_space:
            beta_y = self._solve_beta_for_y_group(int(y), p_test, g_test)
            C_test = self._C_hat(p_test, beta_y)
            if int(y) in C_test:
                out.append(int(y))
        return out
