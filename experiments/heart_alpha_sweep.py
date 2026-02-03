import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.rac.base_rac import BaseRAC
from src.rac.group_rac import GroupConditionalRAC


# ============================================================
# Medical decision utilities (miss disease is worse than overtreat)
# ============================================================

def utility(a: int, y: int) -> float:
    # a=0: "no treat", a=1: "treat"
    # y=1: disease present, y=0: no disease
    if a == 0 and y == 0: return 0.0
    if a == 0 and y == 1: return -8.0   # false negative (miss disease)
    if a == 1 and y == 0: return -2.0   # false positive (unnecessary treatment)
    if a == 1 and y == 1: return 5.0
    raise ValueError("Invalid (a,y)")

def worst_case_action(C, y_space=(0, 1)):
    if C is None or len(C) == 0:
        C = y_space
    return max([0, 1], key=lambda a: min(utility(a, int(y)) for y in C))

def cvar_mean(u: np.ndarray, tail_alpha: float = 0.05) -> float:
    k = max(1, int(np.ceil(tail_alpha * len(u))))
    return float(np.sort(u)[:k].mean())

def eval_all(model, X, Y, G, y_space=(0, 1), tail_alpha=0.05):
    covered = np.array([int(Y[i] in model.predict_set(X[i])) for i in range(len(Y))])

    utils = []
    empty = 0
    for i in range(len(Y)):
        C = model.predict_set(X[i])
        if len(C) == 0:
            empty += 1
        a = worst_case_action(C, y_space=y_space)
        utils.append(utility(a, int(Y[i])))

    utils = np.array(utils, dtype=float)

    out = {
        "mis": float(1.0 - covered.mean()),
        "set_size": float(np.mean([len(model.predict_set(x)) for x in X])),
        "meanU": float(utils.mean()),
        "cvar5": cvar_mean(utils, tail_alpha),
        "empty_rate": float(empty / len(Y)),
    }
    for g in np.unique(G):
        idx = np.where(G == g)[0]
        out[f"mis_g{g}"] = float(1.0 - covered[idx].mean())
        out[f"cvar5_g{g}"] = cvar_mean(utils[idx], tail_alpha)
    return out


# ============================================================
# Load Heart Disease dataset (OpenML id=53, "heart-statlog")
# ============================================================

def load_heart_openml():
    data = fetch_openml(data_id=53, as_frame=True)   # heart-statlog
    X_df = data.data.copy()
    y = data.target

    # Convert y to {0,1}
    if y.dtype.name == "category" or y.dtype == object:
        y = y.astype(str)

    # Common cases:
    # - labels are "1"/"2" -> treat "2" as positive
    # - labels are "0"/"1" -> 1 positive
    y_vals = pd.Series(y).astype(str).values
    uniq = sorted(set(y_vals.tolist()))

    if set(uniq) == {"1", "2"}:
        Y = (y_vals == "2").astype(int)
    elif set(uniq) == {"0", "1"}:
        Y = (y_vals == "1").astype(int)
    else:
        # fallback: map max label to 1, others to 0
        Y = (y_vals == max(uniq)).astype(int)

    # sex column should exist in this dataset (often 0/1)
    if "sex" not in X_df.columns:
        raise RuntimeError(f"Expected 'sex' column in features, got columns: {list(X_df.columns)}")

    # Group by sex
    G = pd.Series(X_df["sex"]).astype(int).values

    # Use ALL features (including sex) for prediction in medical setting
    X_df = X_df.copy()

    # One-hot encode categorical cols if any, then scale
    X_enc = pd.get_dummies(X_df, drop_first=False)
    X = X_enc.values.astype(float)
    X = StandardScaler().fit_transform(X)

    return X, Y.astype(int), G


# ============================================================
# Main: alpha sweep + per-group tail plot
# ============================================================

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    np.random.seed(0)

    X, Y, G = load_heart_openml()

    X_train, X_test, Y_train, Y_test, G_train, G_test = train_test_split(
        X, Y, G, test_size=0.4, random_state=0, stratify=Y
    )

    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, Y_train)

    # append group bit for GroupConditionalRAC grouping
    X_train_aug = np.concatenate([X_train, G_train.reshape(-1, 1)], axis=1)
    X_test_aug  = np.concatenate([X_test,  G_test.reshape(-1, 1)], axis=1)

    def score_fn(x, y):
        p = clf.predict_proba(x[:-1].reshape(1, -1))[0]
        return 1.0 - p[int(y)]

    group_fn = lambda x: int(x[-1])
    y_space = [0, 1]

    alphas = [0.05, 0.10, 0.15, 0.20]
    rows = []

    for alpha in alphas:
        rac = BaseRAC(alpha, y_space, score_fn)
        rac.fit(X_train_aug, Y_train)

        gc = GroupConditionalRAC(alpha, group_fn, y_space, score_fn)
        gc.fit(X_train_aug, Y_train)

        r1 = eval_all(rac, X_test_aug, Y_test, G_test)
        r2 = eval_all(gc,  X_test_aug, Y_test, G_test)
        rows.append((alpha, r1, r2))

        print(f"\n=== Heart Disease (OpenML-53) | alpha={alpha} ===")
        print("Marginal:", r1)
        print("GroupRAC:", r2)

    # Plot: per-group tail utility vs alpha
    al = [a for a, _, _ in rows]
    plt.figure()
    plt.plot(al, [r["cvar5_g0"] for _, r, _ in rows], marker="o", linewidth=2, label="Marginal g0")
    plt.plot(al, [r["cvar5_g1"] for _, r, _ in rows], marker="o", linewidth=2, label="Marginal g1")
    plt.plot(al, [g["cvar5_g0"] for _, _, g in rows], marker="o", linewidth=2, label="GroupRAC g0")
    plt.plot(al, [g["cvar5_g1"] for _, _, g in rows], marker="o", linewidth=2, label="GroupRAC g1")
    plt.xlabel("alpha")
    plt.ylabel("CVaR 5% utility")
    plt.title("Heart Disease: Per-Group Tail Utility vs alpha (group=sex)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/heart_tail_utility_per_group.png", dpi=200)

    print("\nSaved plot: plots/heart_tail_utility_per_group.png")
