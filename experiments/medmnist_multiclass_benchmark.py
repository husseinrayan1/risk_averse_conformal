import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import medmnist
from medmnist import INFO

from src.rac.base_rac import BaseRAC
from src.rac.group_rac import GroupConditionalRAC
from torchvision import transforms

# ============================================================
# Repro
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Simple CNN for 28x28 images
# ============================================================
class SmallCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28->14
        x = self.pool(F.relu(self.conv2(x)))  # 14->7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ============================================================
# Utility + decision rule (multi-action, multi-class)
# ============================================================
def build_action_buckets_from_freq(y_train: np.ndarray, K: int, A: int = 4):
    """
    Build A buckets (actions) by sorting classes by frequency and splitting into A groups.
    This yields a reproducible mapping for multi-class labels -> action "protocols".
    """
    counts = np.bincount(y_train, minlength=K).astype(float)
    order = np.argsort(-counts)  # most frequent first

    buckets = [[] for _ in range(A)]
    for i, cls in enumerate(order):
        buckets[i % A].append(int(cls))
    # convert to sets
    return [set(b) for b in buckets]

def make_utility_matrix(K: int, action_buckets, miss_penalty=-8.0, wrong_treat=-3.0, correct=5.0, no_action_wrong=-10.0):
    """
    A=4 actions; each action corresponds to a "protocol" bucket (subset of labels).
    Utility:
      - If true label y is in action bucket: +correct
      - Else:
         - if action=0 (No action): no_action_wrong (large negative)
         - otherwise: wrong_treat (moderate negative)
    Special case: If y in bucket0 (No action bucket), then no-action is fine: +correct.
    """
    A = len(action_buckets)
    U = np.zeros((A, K), dtype=float)
    for a in range(A):
        for y in range(K):
            if y in action_buckets[a]:
                U[a, y] = correct
            else:
                if a == 0:
                    U[a, y] = no_action_wrong
                else:
                    U[a, y] = wrong_treat
    return U

def maxmin_action_from_set(C, U: np.ndarray, y_space):
    """
    a*(x) = argmax_a min_{y in C} U[a,y]
    If C empty, treat as full uncertainty over y_space.
    """
    if C is None or len(C) == 0:
        C = y_space
    best_a = None
    best_val = None
    for a in range(U.shape[0]):
        v = min(U[a, int(y)] for y in C)
        if best_val is None or v > best_val:
            best_val = v
            best_a = a
    return int(best_a)


def cvar_mean(u: np.ndarray, tail_alpha=0.10) -> float:
    k = max(1, int(np.ceil(tail_alpha * len(u))))
    return float(np.sort(u)[:k].mean())


# ============================================================
# Prediction set baselines (use probs stored in x[:K])
# ============================================================
class LAC_CP:
    """
    Split conformal with score s(x,y)=1-p_y.
    Calibration: q = quantile(scores_cal, 1-alpha) with conservative index.
    Set: {y: 1 - p_y <= q} == {p_y >= 1-q}
    """
    def __init__(self, alpha: float, K: int):
        self.alpha = alpha
        self.K = K
        self.q = None

    def fit(self, P_cal: np.ndarray, y_cal: np.ndarray):
        scores = 1.0 - P_cal[np.arange(len(y_cal)), y_cal]
        n = len(scores)
        # conformal quantile (ceil((n+1)(1-alpha))/n)
        k = int(math.ceil((n + 1) * (1 - self.alpha))) - 1
        k = min(max(k, 0), n - 1)
        self.q = float(np.sort(scores)[k])

    def predict_set(self, x):
        p = np.array(x[:self.K], dtype=float)
        return [i for i in range(self.K) if (1.0 - p[i]) <= self.q]


class APS_CP:
    """
    Adaptive Prediction Sets (APS).
    Score for true label: cumulative probability up to and including true label
    when probabilities are sorted descending.
    Calibration: q = quantile(scores_cal, 1-alpha)
    Set: smallest top-prob set whose cumulative >= q
    """
    def __init__(self, alpha: float, K: int):
        self.alpha = alpha
        self.K = K
        self.q = None

    @staticmethod
    def _cum_score_for_true(p: np.ndarray, y_true: int) -> float:
        order = np.argsort(-p)
        cum = 0.0
        for j, cls in enumerate(order):
            cum += float(p[cls])
            if int(cls) == int(y_true):
                return cum
        return 1.0

    def fit(self, P_cal: np.ndarray, y_cal: np.ndarray):
        scores = np.array([self._cum_score_for_true(P_cal[i], int(y_cal[i])) for i in range(len(y_cal))], dtype=float)
        n = len(scores)
        k = int(math.ceil((n + 1) * (1 - self.alpha))) - 1
        k = min(max(k, 0), n - 1)
        self.q = float(np.sort(scores)[k])

    def predict_set(self, x):
        p = np.array(x[:self.K], dtype=float)
        order = np.argsort(-p)
        cum = 0.0
        S = []
        for cls in order:
            S.append(int(cls))
            cum += float(p[cls])
            if cum >= self.q:
                break
        return S


class HPS:
    """
    Highest-Probability Set baseline: include top probs until cumulative >= 1-alpha.
    (No calibration; weak baseline but common.)
    """
    def __init__(self, alpha: float, K: int):
        self.alpha = alpha
        self.K = K

    def fit(self, P_cal, y_cal):
        return

    def predict_set(self, x):
        p = np.array(x[:self.K], dtype=float)
        order = np.argsort(-p)
        cum = 0.0
        S = []
        for cls in order:
            S.append(int(cls))
            cum += float(p[cls])
            if cum >= (1.0 - self.alpha):
                break
        return S


# ============================================================
# Evaluation
# ============================================================
def eval_method(method, X_test: np.ndarray, y_test: np.ndarray, g_test: np.ndarray, U: np.ndarray, K: int):
    y_space = list(range(K))
    covered = np.zeros(len(y_test), dtype=int)
    set_sizes = np.zeros(len(y_test), dtype=float)
    utils = np.zeros(len(y_test), dtype=float)

    empties = 0
    for i in range(len(y_test)):
        C = method.predict_set(X_test[i])
        set_sizes[i] = len(C)
        if len(C) == 0:
            empties += 1
        covered[i] = int(int(y_test[i]) in C)
        a = maxmin_action_from_set(C, U, y_space)
        utils[i] = U[a, int(y_test[i])]

    out = {
        "mis": float(1.0 - covered.mean()),
        "set_size": float(set_sizes.mean()),
        "meanU": float(utils.mean()),
        "cvar10": cvar_mean(utils, 0.10),
        "empty_rate": float(empties / len(y_test)),
    }

    # group metrics (2 groups)
    for gv in sorted(np.unique(g_test).tolist()):
        idx = np.where(g_test == gv)[0]
        if len(idx) == 0:
            continue
        out[f"mis_g{gv}"] = float(1.0 - covered[idx].mean())
        out[f"cvar10_g{gv}"] = cvar_mean(utils[idx], 0.10)

    out["worst_mis_g"] = float(max(out.get("mis_g0", 0.0), out.get("mis_g1", 0.0)))
    out["worst_cvar10_g"] = float(min(out.get("cvar10_g0", 1e9), out.get("cvar10_g1", 1e9)))
    return out


# ============================================================
# Main benchmark
# ============================================================
if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

    # -------- Choose dataset (multi-class)
    # Good defaults:
    # - pathmnist: 9 classes, RGB
    # - bloodmnist: 8 classes, RGB
    dataset_flag = os.environ.get("MEDMNIST_DATASET", "pathmnist").lower()

    info = INFO[dataset_flag]
    DataClass = getattr(medmnist, info["python_class"])
    K = len(info["label"]) 
    in_ch = int(info["n_channels"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dataset: {dataset_flag} | K={K}, channels={in_ch} | device={device}")

    # -------- Settings
    seeds = [0, 1, 2]  # paper-scale later: bump to 10+
    alphas = [0.05, 0.10, 0.15, 0.20]
    batch_size = 256
    epochs = 3         # increase to 10–20 for stronger model
    lr = 1e-3

    # We will do: Train -> use "val" as calibration -> test for evaluation
    # MedMNIST provides train/val/test splits.
    all_results = {a: {"RAC": [], "GroupRAC": [], "LAC": [], "APS": [], "HPS": []} for a in alphas}

    for seed in seeds:
        set_seed(seed)
        tfm = transforms.ToTensor()

        train_set = DataClass(split="train", download=True, transform=tfm)
        val_set = DataClass(split="val", download=True, transform=tfm)
        test_set = DataClass(split="test", download=True, transform=tfm)

        # loaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2)

        # -------- Train CNN
        model = SmallCNN(in_ch=in_ch, num_classes=K).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for ep in range(epochs):
            pbar = tqdm(train_loader, desc=f"seed={seed} train ep={ep+1}/{epochs}", leave=False)
            for x, y in pbar:
                # x: (B, C, 28, 28), y: (B, 1) or (B,)
                x = x.float().to(device)
                y = y.squeeze().long().to(device)

                logits = model(x)
                loss = F.cross_entropy(logits, y)

                opt.zero_grad()
                loss.backward()
                opt.step()
                pbar.set_postfix({"loss": float(loss.item())})

        # -------- Helper: get probs + labels + group
        def collect_probs(loader):
            model.eval()
            P_list = []
            y_list = []
            g_list = []
            with torch.no_grad():
                for x, y in loader:
                    x = x.float().to(device)
                    y = y.squeeze().long().cpu().numpy()
                    logits = model(x)
                    p = F.softmax(logits, dim=1).cpu().numpy()

                    # group: intensity-based (creates heterogeneous subpops without metadata)
                    # group = 1 if mean pixel > median (computed later)
                    intensity = x.mean(dim=(1,2,3)).cpu().numpy()  # per-sample mean

                    P_list.append(p)
                    y_list.append(y)
                    g_list.append(intensity)
            P = np.concatenate(P_list, axis=0)
            y = np.concatenate(y_list, axis=0).astype(int)
            inten = np.concatenate(g_list, axis=0).astype(float)
            return P, y, inten

        P_tr, y_tr, inten_tr = collect_probs(train_loader)
        P_cal, y_cal, inten_cal = collect_probs(val_loader)
        P_te, y_te, inten_te = collect_probs(test_loader)

        # define groups by median intensity on calibration split
        med = float(np.median(inten_cal))
        g_cal = (inten_cal > med).astype(int)
        g_te  = (inten_te > med).astype(int)

        # Build action buckets from training label frequency (reproducible)
        action_buckets = build_action_buckets_from_freq(y_tr, K, A=4)
        U = make_utility_matrix(K, action_buckets, correct=5.0, wrong_treat=-3.0, no_action_wrong=-10.0)

        # Build X matrices where x[:K] are probs, and x[-1] is group id
        X_cal = np.concatenate([P_cal, g_cal.reshape(-1, 1)], axis=1)
        X_te  = np.concatenate([P_te,  g_te.reshape(-1, 1)], axis=1)

        # Score fn uses probs directly
        def score_fn(x, y):
            return 1.0 - float(x[int(y)])  # x holds probs in first K dims

        group_fn = lambda x: int(x[-1])
        y_space = list(range(K))

        for alpha in alphas:
            # ---------- RAC (your implementation)
            rac = BaseRAC(alpha, y_space, score_fn)
            rac.fit(X_cal, y_cal)

            # ---------- GroupRAC (your extension)
            grac = GroupConditionalRAC(alpha, group_fn, y_space, score_fn)
            grac.fit(X_cal, y_cal)

            # ---------- Baselines
            lac = LAC_CP(alpha, K); lac.fit(P_cal, y_cal)
            aps = APS_CP(alpha, K); aps.fit(P_cal, y_cal)
            hps = HPS(alpha, K);    hps.fit(P_cal, y_cal)

            # Evaluate
            all_results[alpha]["RAC"].append(eval_method(rac,  X_te, y_te, g_te, U, K))
            all_results[alpha]["GroupRAC"].append(eval_method(grac, X_te, y_te, g_te, U, K))
            all_results[alpha]["LAC"].append(eval_method(lac,  X_te, y_te, g_te, U, K))
            all_results[alpha]["APS"].append(eval_method(aps,  X_te, y_te, g_te, U, K))
            all_results[alpha]["HPS"].append(eval_method(hps,  X_te, y_te, g_te, U, K))

        print(f"Done seed={seed}")

    # -------- Aggregate across seeds
    def agg(metric_list):
        # metric_list: list of dicts
        keys = metric_list[0].keys()
        out = {}
        for k in keys:
            vals = [m[k] for m in metric_list if k in m]
            out[k] = float(np.mean(vals))
        return out

    summary = {a: {} for a in alphas}
    for alpha in alphas:
        for name, lst in all_results[alpha].items():
            summary[alpha][name] = agg(lst)

    # Print concise summary
    for alpha in alphas:
        print(f"\n=== MedMNIST {dataset_flag} | alpha={alpha} | avg over seeds={len(seeds)} ===")
        for name in ["RAC", "GroupRAC", "LAC", "APS", "HPS"]:
            print(name, summary[alpha][name])

    # -------- Plot helpers
    def plot_metric(metric_key: str, title: str, ylabel: str, fname: str):
        plt.figure()
        for name in ["RAC", "GroupRAC", "LAC", "APS", "HPS"]:
            ys = [summary[a][name][metric_key] for a in alphas]
            plt.plot(alphas, ys, marker="o", linewidth=2, label=name)
        plt.xlabel("alpha")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fname, dpi=200)

    plot_metric("meanU",
                f"{dataset_flag}: Mean Utility vs alpha (A=4 actions)",
                "mean utility",
                f"plots/{dataset_flag}_meanU_vs_alpha.png")

    plot_metric("cvar10",
                f"{dataset_flag}: CVaR10 Utility vs alpha (overall)",
                "CVaR 10% utility",
                f"plots/{dataset_flag}_cvar10_vs_alpha.png")

    plot_metric("set_size",
                f"{dataset_flag}: Avg Prediction Set Size vs alpha",
                "avg |C(x)|",
                f"plots/{dataset_flag}_setsize_vs_alpha.png")

    plot_metric("mis",
                f"{dataset_flag}: Miscoverage vs alpha",
                "miscoverage",
                f"plots/{dataset_flag}_mis_vs_alpha.png")

    plot_metric("worst_mis_g",
                f"{dataset_flag}: Worst-Group Miscoverage vs alpha (2 groups by intensity)",
                "worst-group miscoverage",
                f"plots/{dataset_flag}_worstgroup_mis_vs_alpha.png")

    print("\nSaved plots:")
    print(f" - plots/{dataset_flag}_meanU_vs_alpha.png")
    print(f" - plots/{dataset_flag}_cvar10_vs_alpha.png")
    print(f" - plots/{dataset_flag}_setsize_vs_alpha.png")
    print(f" - plots/{dataset_flag}_mis_vs_alpha.png")
    print(f" - plots/{dataset_flag}_worstgroup_mis_vs_alpha.png")
    print("\nTip: change dataset via env var, e.g. MEDMNIST_DATASET=bloodmnist")
