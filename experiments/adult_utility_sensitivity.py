import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.rac.base_rac import BaseRAC
from src.rac.group_rac import GroupConditionalRAC


def cvar_mean(u: np.ndarray, tail_alpha=0.05) -> float:
    k = max(1, int(np.ceil(tail_alpha * len(u))))
    return float(np.sort(u)[:k].mean())


def load_adult_sex():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    df = pd.read_csv(url, header=None)

    df.columns = [
        "age","workclass","fnlwgt","education","education_num","marital_status",
        "occupation","relationship","race","sex","capital_gain","capital_loss",
        "hours_per_week","native_country","income"
    ]

    df = df.replace(" ?", np.nan).dropna()

    Y = (df["income"] == " >50K").astype(int).values.astype(int)
    G = (df["sex"] == " Male").astype(int).values.astype(int)

    X = pd.get_dummies(df.drop(["income", "sex"], axis=1))
    X = StandardScaler().fit_transform(X.values.astype(float))
    return X, Y, G


def run_once(alpha, c_fp, c_fn, seed=0):
    np.random.seed(seed)
    X, Y, G = load_adult_sex()

    X_train, X_test, Y_train, Y_test, G_train, G_test = train_test_split(
        X, Y, G, test_size=0.4, random_state=seed, stratify=Y
    )

    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_train, Y_train)

    X_train_aug = np.concatenate([X_train, G_train.reshape(-1,1)], axis=1)
    X_test_aug  = np.concatenate([X_test,  G_test.reshape(-1,1)], axis=1)

    def score_fn(x, y):
        p = clf.predict_proba(x[:-1].reshape(1,-1))[0]
        return 1.0 - p[int(y)]

    group_fn = lambda x: int(x[-1])
    y_space = [0,1]

    def utility(a, y):
        # deny=0, approve=1
        if a == 0 and y == 0: return 0.0
        if a == 0 and y == 1: return -float(c_fn)   # false negative cost
        if a == 1 and y == 0: return -float(c_fp)   # false positive cost
        if a == 1 and y == 1: return 5.0
        raise ValueError

    def worst_case_action(C):
        if C is None or len(C) == 0:
            C = y_space
        return max([0,1], key=lambda a: min(utility(a, int(y)) for y in C))

    def eval_worst_group_cvar(model):
        utils = []
        for i in range(len(Y_test)):
            C = model.predict_set(X_test_aug[i])
            a = worst_case_action(C)
            utils.append(utility(a, int(Y_test[i])))
        utils = np.array(utils, dtype=float)

        worst = None
        for g in np.unique(G_test):
            idx = np.where(G_test == g)[0]
            val = cvar_mean(utils[idx], 0.05)
            worst = val if worst is None else min(worst, val)  # lower is worse
        return float(worst)

    rac = BaseRAC(alpha, y_space, score_fn); rac.fit(X_train_aug, Y_train)
    gc  = GroupConditionalRAC(alpha, group_fn, y_space, score_fn); gc.fit(X_train_aug, Y_train)

    return eval_worst_group_cvar(rac), eval_worst_group_cvar(gc)


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

    alpha = 0.10
    c_fp_grid = [2, 5, 8, 12]
    c_fn_grid = [2, 5, 8, 12]

    # heatmaps: improvement = GroupRAC - Marginal (higher is better since utilities are negative in tail)
    improvement = np.zeros((len(c_fn_grid), len(c_fp_grid)))

    for i, c_fn in enumerate(c_fn_grid):
        for j, c_fp in enumerate(c_fp_grid):
            w_marg, w_group = run_once(alpha, c_fp=c_fp, c_fn=c_fn, seed=0)
            improvement[i, j] = w_group - w_marg
            print(f"c_fn={c_fn:>2}, c_fp={c_fp:>2} | worst-CVaR5: marginal={w_marg:.3f}, group={w_group:.3f}, improve={improvement[i,j]:.3f}")

    plt.figure()
    plt.imshow(improvement, aspect="auto")
    plt.xticks(range(len(c_fp_grid)), [str(x) for x in c_fp_grid])
    plt.yticks(range(len(c_fn_grid)), [str(x) for x in c_fn_grid])
    plt.xlabel("false positive cost (c_fp)")
    plt.ylabel("false negative cost (c_fn)")
    plt.title("Adult: Worst-Group CVaR5 Improvement (GroupRAC - Marginal) at alpha=0.10")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("plots/adult_utility_sensitivity_improvement.png", dpi=200)
