import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.rac.base_rac import BaseRAC
from src.rac.group_rac import GroupConditionalRAC


# ============================================================
# Decision + utility
# ============================================================

def utility(a: int, y: int) -> float:
    # deny=0, approve=1 ; y=0 <=50K, y=1 >50K
    if a == 0 and y == 0: return 0.0
    if a == 0 and y == 1: return -2.0
    if a == 1 and y == 0: return -5.0
    if a == 1 and y == 1: return 5.0
    raise ValueError("Invalid (a,y)")

def worst_case_action(C, y_space=(0, 1)):
    # If empty set, fall back to full label space (max uncertainty)
    if C is None or len(C) == 0:
        C = y_space
    return max([0, 1], key=lambda a: min(utility(a, int(y)) for y in C))

def cvar_mean(u: np.ndarray, tail_alpha: float = 0.05) -> float:
    k = max(1, int(np.ceil(tail_alpha * len(u))))
    return float(np.sort(u)[:k].mean())


# ============================================================
# Data: UCI Adult with intersectional groups (sex × race)
# ============================================================

def load_adult_intersectional():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    df = pd.read_csv(url, header=None)

    df.columns = [
        "age","workclass","fnlwgt","education","education_num","marital_status",
        "occupation","relationship","race","sex","capital_gain","capital_loss",
        "hours_per_week","native_country","income"
    ]

    df = df.replace(" ?", np.nan).dropna()

    Y = (df["income"] == " >50K").astype(int).values.astype(int)

    # intersectional group string -> integer id
    g_str = (df["sex"].astype(str) + "|" + df["race"].astype(str)).values
    G, group_names = pd.factorize(g_str)  # G in {0..K-1}

    # predictor features exclude sex/race to isolate calibration effect
    X = pd.get_dummies(df.drop(["income", "sex", "race"], axis=1))
    X = StandardScaler().fit_transform(X.values.astype(float))

    return X, Y, G, group_names


# ============================================================
# Evaluation: overall + worst-group + avg-worst-5-groups
# ============================================================

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

    # group-wise arrays
    mis_g = []
    cvar_g = []
    for g in np.unique(G):
        idx = np.where(G == g)[0]
        mis_g.append(float(1.0 - covered[idx].mean()))
        cvar_g.append(cvar_mean(utils[idx], tail_alpha))

    mis_g = np.array(mis_g)
    cvar_g = np.array(cvar_g)

    k = min(5, len(cvar_g))
    avg_worst5 = float(np.sort(cvar_g)[:k].mean())  # mean of worst 5 groups

    out = {
        "mis": float(1.0 - covered.mean()),
        "set_size": float(np.mean([len(model.predict_set(x)) for x in X])),
        "meanU": float(utils.mean()),
        "cvar5": cvar_mean(utils, tail_alpha),
        "empty_rate": float(empty / len(Y)),
        "worst_mis_g": float(np.max(mis_g)),
        "worst_cvar5_g": float(np.min(cvar_g)),          # single worst group
        "avg_worst5_cvar5_g": float(avg_worst5),         # more stable metric
        "num_groups": int(len(np.unique(G))),
    }
    return out


# ============================================================
# Main: alpha sweep + better plots
# ============================================================

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    np.random.seed(0)

    X, Y, G, group_names = load_adult_intersectional()

    X_train, X_test, Y_train, Y_test, G_train, G_test = train_test_split(
        X, Y, G, test_size=0.4, random_state=0, stratify=Y
    )

    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_train, Y_train)

    # append group id as last feature for GroupConditionalRAC grouping
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

        print(f"\n=== Adult intersectional (sex×race) | alpha={alpha} ===")
        print("Marginal:", r1)
        print("GroupRAC:", r2)

    al = [a for a, _, _ in rows]

    # Plot 1: overall CVaR5 vs alpha
    plt.figure()
    plt.plot(al, [r["cvar5"] for _, r, _ in rows], marker="o", linewidth=2, alpha=0.9, label="Marginal overall CVaR5", zorder=2)
    plt.plot(al, [g["cvar5"] for _, _, g in rows], marker="o", linewidth=2, alpha=0.9, label="GroupRAC overall CVaR5", zorder=3)
    plt.xlabel("alpha")
    plt.ylabel("CVaR 5% utility (overall)")
    plt.title("Adult (sex×race): Overall Tail Utility vs alpha")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/adult_intersectional_overall_cvar5.png", dpi=200)

    # Plot 2: worst-group miscoverage vs alpha
    plt.figure()
    plt.plot(al, [r["worst_mis_g"] for _, r, _ in rows], marker="o", linewidth=2, alpha=0.9, label="Marginal worst-group miscoverage", zorder=2)
    plt.plot(al, [g["worst_mis_g"] for _, _, g in rows], marker="o", linewidth=2, alpha=0.9, label="GroupRAC worst-group miscoverage", zorder=3)
    plt.xlabel("alpha")
    plt.ylabel("Worst-group miscoverage")
    plt.title("Adult (sex×race): Worst-Group Miscoverage vs alpha")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/adult_intersectional_worstgroup_miscoverage.png", dpi=200)

    # Plot 3: avg of worst-5 groups CVaR5 vs alpha (best intersectional plot)
    plt.figure()
    plt.plot(al, [r["avg_worst5_cvar5_g"] for _, r, _ in rows], marker="o", linewidth=2, alpha=0.9, label="Marginal avg worst-5 groups CVaR5", zorder=2)
    plt.plot(al, [g["avg_worst5_cvar5_g"] for _, _, g in rows], marker="o", linewidth=2, alpha=0.9, label="GroupRAC avg worst-5 groups CVaR5", zorder=3)
    plt.xlabel("alpha")
    plt.ylabel("Avg worst-5 groups CVaR 5% utility")
    plt.title("Adult (sex×race): Avg Worst-5 Groups Tail Utility vs alpha")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/adult_intersectional_avg_worst5_cvar5.png", dpi=200)

    print("\nSaved plots:")
    print(" - plots/adult_intersectional_overall_cvar5.png")
    print(" - plots/adult_intersectional_worstgroup_miscoverage.png")
    print(" - plots/adult_intersectional_avg_worst5_cvar5.png")
