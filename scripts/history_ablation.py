"""历史特征消融 H1–H5：究竟是历史投票没信息，还是我们的表达方式太简单？

设计
----
每折只训练一次视觉头（差分头，60 epoch），五档历史表示共用同一组视觉 logits，
因此各档之间的差异只来自历史特征本身。

H1  净胜分 q=(w-l)/(n+8)                      原稿使用
H2  Davidson 潜在强度 θ                       对手与平局同时建模
H3  θ + 后验标准差                            加入可靠性
H4  H3 + log(1+n) + 平局率                    加入证据量与判断倾向
H5  H4 + 对手平均强度                          加入赛程强度

每档都在验证城跨城票上拟合 [1,d]→3 logits 的线性映射，再在验证票上选 β，
最后在测试城跨城票上评估。全部只用训练票计算历史特征，无泄漏。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from converge_heads import Head
from anchor_history_fusion import fit_beta, logloss

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\ablation")
LABELS = {"left": 0, "equal": 1, "right": 2}
SHRINK = 8.0


# ---------------------------------------------------------------- 基础计数

def base_counts(ia, ib, y, n_img):
    """净胜分与其组成，全部只用训练票。"""
    left, right, tie = (y == 0), (y == 2), (y == 1)
    n = np.bincount(ia, minlength=n_img) + np.bincount(ib, minlength=n_img)
    w = np.bincount(ia[left], minlength=n_img) + np.bincount(ib[right], minlength=n_img)
    l = np.bincount(ib[left], minlength=n_img) + np.bincount(ia[right], minlength=n_img)
    t = np.bincount(ia[tie], minlength=n_img) + np.bincount(ib[tie], minlength=n_img)
    w = w + 0.5 * t
    l = l + 0.5 * t
    q = (w - l) / (n + SHRINK)
    tie_rate = t / np.maximum(n, 1)
    return dict(n=n, w=w, l=l, t=t, q=q, tie_rate=tie_rate)


# ---------------------------------------------------------------- Davidson

def fit_davidson(ia, ib, y, n_img, l2=5e-2, iters=200):
    """三结果 Davidson MLE。

    数值处理：对数域统一减去 pair 内最大 θ，避免 exp 溢出；
    可识别性：θ 只定义到一个常数，每次迭代后中心化。
    """
    ia_t = torch.tensor(ia, dtype=torch.long)
    ib_t = torch.tensor(ib, dtype=torch.long)
    y_t = torch.tensor(y, dtype=torch.long)

    theta = torch.zeros(n_img, requires_grad=True)
    log_nu = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([theta, log_nu], max_iter=iters,
                            line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        ti, tj = theta[ia_t], theta[ib_t]
        m = torch.maximum(ti, tj).detach()
        ti2, tj2 = ti - m, tj - m
        nu = torch.exp(log_nu)
        log_s = log_nu + (ti2 + tj2) / 2
        s = torch.exp(log_s)
        denom = torch.exp(ti2) + torch.exp(tj2) + s
        logden = torch.log(denom)
        logp = torch.stack([ti2 - logden, log_s - logden, tj2 - logden], 1)
        loss = F.nll_loss(logp, y_t) + l2 * (theta ** 2).mean()
        loss.backward()
        return loss

    for _ in range(6):
        opt.step(closure)
        with torch.no_grad():
            theta -= theta.mean()
            theta.clamp_(-8, 8)
            log_nu.clamp_(-6, 6)
    th = theta.detach().numpy().astype(np.float64)

    with torch.no_grad():
        ti, tj = theta[ia_t], theta[ib_t]
        m = torch.maximum(ti, tj)
        ti2, tj2 = ti - m, tj - m
        nu = torch.exp(log_nu)
        s = nu * torch.exp((ti2 + tj2) / 2)
        denom = torch.exp(ti2) + torch.exp(tj2) + s
        p_l, p_t, p_r = torch.exp(ti2) / denom, s / denom, torch.exp(tj2) / denom
        gi = torch.zeros(n_img)
        gi.index_add_(0, ia_t, p_l * (1 - p_l) + p_t * (0.5 - p_t) ** 2)
        gi.index_add_(0, ib_t, p_r * (1 - p_r) + p_t * (0.5 - p_t) ** 2)
        sd = 1.0 / torch.sqrt(gi + 1e-4)
    return th, sd.clamp(max=10.).numpy().astype(np.float64), float(torch.exp(log_nu))


def opponent_strength(ia, ib, y, theta, n_img):
    """每个图像对手的平均 θ。"""
    left, right, tie = (y == 0), (y == 2), (y == 1)
    tot = np.zeros(n_img)
    cnt = np.zeros(n_img)
    # a 的对手是 b，b 的对手是 a
    np.add.at(tot, ia, theta[ib]); np.add.at(cnt, ia, 1)
    np.add.at(tot, ib, theta[ia]); np.add.at(cnt, ib, 1)
    return tot / np.maximum(cnt, 1)


# ---------------------------------------------------------------- 特征表

def build_variants(cnt, theta, sd, opp):
    v = {}
    v["H1_netwin"] = [cnt["q"]]
    v["H2_davidson"] = [theta]
    v["H3_mean_sd"] = [theta, sd]
    v["H4_mean_sd_n_tie"] = [theta, sd, np.log1p(cnt["n"]), cnt["tie_rate"]]
    v["H5_plus_opponent"] = [theta, sd, np.log1p(cnt["n"]), cnt["tie_rate"], opp]
    return {k: np.stack(x, 1) for k, x in v.items()}


def fit_hist_multi(D: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    X = torch.tensor(np.concatenate([np.ones((len(D), 1)), D], 1),
                     dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    p = torch.zeros(X.shape[1], 3, requires_grad=True)
    opt = torch.optim.LBFGS([p], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X @ p, yt) + l2 * (p ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return p.detach().numpy()


def hist_logits_multi(Q, ia, ib, params):
    D = Q[ia] - Q[ib]
    X = np.concatenate([np.ones((len(D), 1)), D], 1).astype(np.float32)
    return X @ params


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", type=Path,
                    default=DATA / "features_convnext_tiny.npz")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(DATA / "pairs.parquet")
    folds = json.loads((DATA / "splits" / "logo_folds.json").read_text())
    z = np.load(args.features)
    idx = {k: i for i, k in enumerate(z["ids"].tolist())}
    X = torch.from_numpy(z["features"].astype(np.float32)).to(device)
    n_img = X.shape[0]

    df = df[df["left"].isin(idx) & df["right"].isin(idx)].reset_index(drop=True)
    df["_ia"] = df["left"].map(idx).astype(np.int32)
    df["_ib"] = df["right"].map(idx).astype(np.int32)
    df["_y"] = df["choice"].map(LABELS).astype(np.int8)
    print(f"epochs={args.epochs} seed={args.seed} features={args.features.stem}")

    rows = []
    t0 = time.time()
    for fi, f in enumerate(folds):
        c, v = f["test_city"], f["val_city"]
        in_t = (df["left_city"] == c) | (df["right_city"] == c)
        in_v = (df["left_city"] == v) | (df["right_city"] == v)
        tr = df[(~in_t) & (~in_v)]
        va_same = df[in_v & df["same_city"] & (~in_t)]
        va_cross = df[(in_v & (~in_t)) & (~df["same_city"])]
        te = df[in_t & (~df["same_city"])]

        torch.manual_seed(args.seed)
        model = Head(dim=int(X.shape[1]), mode="diff").to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        ia = torch.tensor(tr["_ia"].values, device=device)
        ib = torch.tensor(tr["_ib"].values, device=device)
        yy = torch.tensor(tr["_y"].values, dtype=torch.long, device=device)
        va_ia = torch.tensor(va_same["_ia"].values, device=device)
        va_ib = torch.tensor(va_same["_ib"].values, device=device)
        va_y = torch.tensor(va_same["_y"].values, dtype=torch.long, device=device)

        best = (9e9, None)
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(len(ia), device=device)
            for k in range(0, len(ia), 8192):
                j = perm[k:k + 8192]
                loss = F.cross_entropy(model(X[ia[j]], X[ib[j]]), yy[j])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                ce = float(F.cross_entropy(model(X[va_ia], X[va_ib]), va_y))
            if ce < best[0]:
                best = (ce, {k2: t2.detach().clone()
                             for k2, t2 in model.state_dict().items()})
        if best[1]:
            model.load_state_dict(best[1])
        model.eval()

        with torch.no_grad():
            def pred(d):
                return model(X[torch.tensor(d["_ia"].values, device=device)],
                             X[torch.tensor(d["_ib"].values, device=device)]
                             ).cpu().numpy()
            b2_va, b2_te = pred(va_cross), pred(te)

        # 历史特征（只用训练票）
        tr_ia, tr_ib, tr_y = (tr["_ia"].values, tr["_ib"].values, tr["_y"].values)
        cnt = base_counts(tr_ia, tr_ib, tr_y, n_img)
        theta, sd, nu = fit_davidson(tr_ia, tr_ib, tr_y, n_img)
        opp = opponent_strength(tr_ia, tr_ib, tr_y, theta, n_img)
        variants = build_variants(cnt, theta, sd, opp)

        y_te = te["_y"].values
        base = logloss(b2_te, y_te)
        rec = {"fold": c, "n_test": len(te), "A": base, "nu": nu}
        for name, Q in variants.items():
            params = fit_hist_multi(Q[va_cross["_ia"].values] - Q[va_cross["_ib"].values],
                                    va_cross["_y"].values)
            h_va = hist_logits_multi(Q, va_cross["_ia"].values, va_cross["_ib"].values, params)
            h_te = hist_logits_multi(Q, te["_ia"].values, te["_ib"].values, params)
            beta = fit_beta(b2_va, h_va, va_cross["_y"].values)
            rec[name] = logloss(b2_te + beta * h_te, y_te) - base
            rec[name + "_beta"] = beta
        rows.append(rec)
        print(f"[{fi+1:>2}] {c[:16]:<18} A={base:.6f}  " +
              "  ".join(f"{k.split('_')[0]}={rec[k]:+.6f}" for k in variants))

    out = pd.DataFrame(rows)
    tag = f"hist_ablation_e{args.epochs}_s{args.seed}_{args.features.stem}"
    out.to_csv(args.out / f"{tag}.csv", index=False)

    print(f"\n用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"  A（纯视觉）{out['A'].mean():.6f}")
    print(f"  {'变体':<20}{'Δ':>12}{'相对':>10}{'更好折数':>10}")
    for name in [c for c in out.columns if c.startswith("H") and "_beta" not in c]:
        d = out[name]
        print(f"  {name:<20}{d.mean():>+12.6f}"
              f"{-d.mean()/out['A'].mean()*100:>9.3f}%{int((d<0).sum()):>7}/15")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
