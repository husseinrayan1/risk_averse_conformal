import numpy as np
import matplotlib.pyplot as plt

from src.rac.base_rac import BaseRAC
from src.rac.group_rac import GroupConditionalRAC


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_synthetic(n=2000, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))

    # Define groups by sign of first feature
    G = (X[:, 0] >= 0).astype(int)  # 0/1

    # True label model + group-dependent noise
    w = np.array([1.5, -0.5])
    logits = X @ w

    # Group 1 is "harder": more noise (less separable)
    noise = rng.normal(scale=0.5, size=n)
    noise[G == 1] = rng.normal(scale=1.5, size=(G == 1).sum())

    probs = sigmoid(logits + noise)
    Y = (rng.uniform(size=n) < probs).astype(int)

    return X, Y, G


def score_fn_factory(w):
    # score(x, y) = 1 - p(y|x)
    def score(x, y):
        p1 = sigmoid(np.dot(w, x))
        py = p1 if int(y) == 1 else (1.0 - p1)
        return 1.0 - py
    return score


def eval_miscoverage(model, X, Y, G=None):
    covered = np.array([int(Y[i] in model.predict_set(X[i])) for i in range(len(Y))])
    mis = 1.0 - covered.mean()

    if G is None:
        return {"miscoverage": mis, "set_size": np.mean([len(model.predict_set(x)) for x in X])}

    out = {"miscoverage": mis}
    for g in np.unique(G):
        idx = np.where(G == g)[0]
        out[f"miscoverage_g{g}"] = 1.0 - covered[idx].mean()
        out[f"set_size_g{g}"] = np.mean([len(model.predict_set(X[i])) for i in idx])
    return out


if __name__ == "__main__":
    alpha = 0.1
    y_space = [0, 1]

    X, Y, G = make_synthetic(n=4000, seed=0)

    # Split into calibration and test
    n_cal = 1500
    X_cal, Y_cal, G_cal = X[:n_cal], Y[:n_cal], G[:n_cal]
    X_te, Y_te, G_te = X[n_cal:], Y[n_cal:], G[n_cal:]

    # Fixed "model" weights (no training)
    w = np.array([1.0, -0.3])
    score_fn = score_fn_factory(w)

    # Marginal (single beta)
    rac = BaseRAC(alpha=alpha, y_space=y_space, score_fn=score_fn)
    rac.fit(X_cal, Y_cal)

    # Group-conditional (beta per group)
    gc = GroupConditionalRAC(alpha=alpha, group_fn=lambda x: int(x[0] >= 0), y_space=y_space, score_fn=score_fn)
    gc.fit(X_cal, Y_cal)

    print("=== Calibrated betas ===")
    print("marginal beta:", rac.result.beta)
    for g, m in gc.models.items():
        print(f"group {g} beta:", m.result.beta)

    print("\n=== Test evaluation ===")
    print("Marginal RAC:", eval_miscoverage(rac, X_te, Y_te, G_te))
    print("Group-Conditional RAC:", eval_miscoverage(gc, X_te, Y_te, G_te))
    # --- Plot miscoverage per group ---
    res_m = eval_miscoverage(rac, X_te, Y_te, G_te)
    res_gc = eval_miscoverage(gc, X_te, Y_te, G_te)

    groups = ["g0", "g1"]
    mis_m = [float(res_m["miscoverage_g0"]), float(res_m["miscoverage_g1"])]
    mis_gc = [float(res_gc["miscoverage_g0"]), float(res_gc["miscoverage_g1"])]

    x = np.arange(len(groups))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, mis_m, width, label="Marginal RAC")
    plt.bar(x + width/2, mis_gc, width, label="Group-Conditional RAC")
    plt.axhline(alpha, linestyle="--", label="target alpha")

    plt.xticks(x, groups)
    plt.ylabel("Miscoverage")
    plt.title("Miscoverage by Group")
    plt.legend()
    plt.tight_layout()

    # Save + show
    plt.savefig("report_notes/miscoverage_by_group.png", dpi=200)
    plt.show()
