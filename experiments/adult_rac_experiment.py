# experiments/adult_rac_experiment.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.rac.base_rac import BaseRAC
from src.rac.group_rac import GroupConditionalRAC


# ============================================================
# 1) DECISION PROBLEM
# ============================================================

def utility(a: int, y: int) -> float:
    if a == 0 and y == 0:
        return 0.0
    if a == 0 and y == 1:
        return -2.0
    if a == 1 and y == 0:
        return -5.0
    if a == 1 and y == 1:
        return 5.0
    raise ValueError("Invalid (a,y)")


def worst_case_action(C, y_space=(0, 1)):
    if C is None or len(C) == 0:
        C = y_space
    actions = [0, 1]
    return max(actions, key=lambda a: min(utility(a, int(y)) for y in C))


# ============================================================
# 2) DATASET
# ============================================================

def load_adult():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    df = pd.read_csv(url, header=None)

    df.columns = [
        "age","workclass","fnlwgt","education","education_num","marital_status",
        "occupation","relationship","race","sex","capital_gain","capital_loss",
        "hours_per_week","native_country","income"
    ]

    df = df.replace(" ?", np.nan).dropna()

    Y = (df["income"] == " >50K").astype(int).values
    G = (df["sex"] == " Male").astype(int).values

    X = pd.get_dummies(df.drop(["income", "sex"], axis=1))
    X = StandardScaler().fit_transform(X.values.astype(float))

    return X, Y, G


# ============================================================
# 3) EVALUATION
# ============================================================

def cvar_mean(u: np.ndarray, tail_alpha: float = 0.05) -> float:
    k = max(1, int(np.ceil(tail_alpha * len(u))))
    return float(np.sort(u)[:k].mean())


def eval_all(model, X, Y, G, y_space=(0, 1), tail_alpha=0.05):
    covered = np.array([int(Y[i] in model.predict_set(X[i])) for i in range(len(Y))])

    utilities = []
    empty_count = 0
    for i in range(len(Y)):
        C = model.predict_set(X[i])
        if len(C) == 0:
            empty_count += 1
        a = worst_case_action(C, y_space)
        utilities.append(utility(a, int(Y[i])))

    utilities = np.array(utilities)

    out = {
        "mis": float(1.0 - covered.mean()),
        "set_size": float(np.mean([len(model.predict_set(x)) for x in X])),
        "meanU": float(utilities.mean()),
        "cvar5": cvar_mean(utilities),
        "empty_rate": float(empty_count / len(Y)),
    }

    for g in np.unique(G):
        idx = np.where(G == g)[0]
        out[f"mis_g{g}"] = float(1.0 - covered[idx].mean())
        out[f"cvar5_g{g}"] = cvar_mean(utilities[idx])

    return out


# ============================================================
# 4) MAIN: α SWEEP
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)

    X, Y, G = load_adult()

    X_train, X_test, Y_train, Y_test, G_train, G_test = train_test_split(
        X, Y, G, test_size=0.4, random_state=0, stratify=Y
    )

    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_train, Y_train)

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

        res_rac = eval_all(rac, X_test_aug, Y_test, G_test)
        res_gc  = eval_all(gc,  X_test_aug, Y_test, G_test)

        rows.append((alpha, res_rac, res_gc))

        print(f"\n=== alpha = {alpha} ===")
        print("Marginal RAC:", res_rac)
        print("Group RAC:   ", res_gc)

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    al = [r[0] for r in rows]

    # Overall tail utility
    plt.figure()
    plt.plot(al, [r[1]["cvar5"] for r in rows], marker="o", label="Marginal RAC")
    plt.plot(al, [r[2]["cvar5"] for r in rows], marker="o", label="Group RAC")
    plt.xlabel("alpha")
    plt.ylabel("CVaR 5% utility")
    plt.title("Overall Tail Utility vs alpha (Adult)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/adult_tail_utility_overall.png", dpi=200)

    # Per-group tail utility
    plt.figure()
    plt.plot(al, [r[1]["cvar5_g0"] for r in rows], marker="o", label="Marginal g0")
    plt.plot(al, [r[1]["cvar5_g1"] for r in rows], marker="o", label="Marginal g1")
    plt.plot(al, [r[2]["cvar5_g0"] for r in rows], marker="o", label="GroupRAC g0")
    plt.plot(al, [r[2]["cvar5_g1"] for r in rows], marker="o", label="GroupRAC g1")
    plt.xlabel("alpha")
    plt.ylabel("CVaR 5% utility")
    plt.title("Per-Group Tail Utility vs alpha (Adult)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/adult_tail_utility_per_group.png", dpi=200)

    plt.show()
