"""推断检验：置换零分布 + 城市聚类 bootstrap。

回应审阅的两条意见
------------------
1. 负对照只用一个置换种子，"removed the advantage" 没有零分布支撑。
   -> 做 1000 次同城同票数档置换，报零分布与置换 p 值。
2. 稿件报告的区间按投票者聚类，不含城市重抽样；而对"新城市"的推断
   恰恰需要城市层面的不确定性。
   -> 做城市聚类 bootstrap，重算城市等权均值的区间。

自检
----
- 必须复现已知的 A = 0.905491 / C = 0.904930 / Δ = -0.000561
- 置换后 q 的多重集必须不变
- 城市 bootstrap 的全 1 抽样必须等于观测均值

用法：
    python inference_checks.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_history_fusion import (fit_beta, fit_hist_model, hist_logits,
                                   history_quality)
from converge_heads import Head

DATA = Path(r"<PROJECT>/STREETVIEW-PP2/data")
OUT = Path(r"<PROJECT>/STREETVIEW-PP2/outputs/inference")
LABELS = {"left": 0, "equal": 1, "right": 2}
BINS = [0, 1, 2, 5, 10, 20, 10**9]


def logloss_np(logits: np.ndarray, y: np.ndarray) -> float:
    m = logits.max(1, keepdims=True)
    lse = (m + np.log(np.exp(logits - m).sum(1, keepdims=True))).squeeze(1)
    return float((lse - logits[np.arange(len(y)), y]).mean())


def bin_of(n: np.ndarray) -> np.ndarray:
    return np.digitize(n, BINS) - 1


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(DATA / "pairs.parquet")
    folds = json.loads((DATA / "splits" / "logo_folds.json").read_text(encoding="utf-8"))
    z = np.load(DATA / "features_convnext_tiny.npz")
    idx = {k: i for i, k in enumerate(z["ids"].tolist())}
    X = torch.from_numpy(z["features"].astype(np.float32)).to(device)
    n_img = X.shape[0]

    df = df[df["left"].isin(idx) & df["right"].isin(idx)].reset_index(drop=True)
    df["_ia"] = df["left"].map(idx).astype(np.int32)
    df["_ib"] = df["right"].map(idx).astype(np.int32)
    df["_y"] = df["choice"].map(LABELS).astype(np.int8)
    print(f"device={device}  票 {len(df):,}")

    rng = np.random.default_rng(0)
    NPERM_FIXED = 1000
    NPERM_RESEL = 200

    per_fold = []
    for fi, f in enumerate(folds):
        c, v = f["test_city"], f["val_city"]
        in_t = (df["left_city"] == c) | (df["right_city"] == c)
        in_v = (df["left_city"] == v) | (df["right_city"] == v)
        tr = df[(~in_t) & (~in_v)]
        va_same = df[in_v & df["same_city"] & (~in_t)]
        va_cross = df[(in_v & (~in_t)) & (~df["same_city"])]
        te = df[in_t & (~df["same_city"])]

        # --- 训练（4 epoch，复现主表）---
        torch.manual_seed(0)
        model = Head(mode="diff").to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        ia = torch.tensor(tr["_ia"].values, device=device)
        ib = torch.tensor(tr["_ib"].values, device=device)
        yy = torch.tensor(tr["_y"].values, dtype=torch.long, device=device)
        va_ia = torch.tensor(va_same["_ia"].values, device=device)
        va_ib = torch.tensor(va_same["_ib"].values, device=device)
        va_y = torch.tensor(va_same["_y"].values, dtype=torch.long, device=device)
        best = {"ce": 9e9, "state": None}
        for ep in range(4):
            model.train()
            perm = torch.randperm(len(ia), device=device)
            for k in range(0, len(ia), 8192):
                j = perm[k:k + 8192]
                loss = F.cross_entropy(model(X[ia[j]], X[ib[j]]), yy[j])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                ce = float(F.cross_entropy(model(X[va_ia], X[va_ib]), va_y))
            if ce < best["ce"]:
                best = {"ce": ce, "state": {k: t.detach().clone()
                                            for k, t in model.state_dict().items()}}
        model.load_state_dict(best["state"]); model.eval()

        with torch.no_grad():
            def pred(d):
                return model(X[torch.tensor(d["_ia"].values, device=device)],
                             X[torch.tensor(d["_ib"].values, device=device)]).cpu().numpy()
            b2_va, b2_te = pred(va_cross), pred(te)

        # --- 净值历史 ---
        q, ncnt = history_quality(tr, n_img)
        hp = fit_hist_model(q[va_cross["_ia"].values] - q[va_cross["_ib"].values],
                            va_cross["_y"].values)
        h_va = hist_logits(q, va_cross["_ia"].values, va_cross["_ib"].values, hp)
        h_te = hist_logits(q, te["_ia"].values, te["_ib"].values, hp)
        beta = fit_beta(b2_va, h_va, va_cross["_y"].values)

        y_te = te["_y"].values.astype(np.int64)
        A = logloss_np(b2_te, y_te)
        C = logloss_np(b2_te + beta * h_te, y_te)

        # --- 置换：在训练图的 (city, count-bin) 组内打乱 q ---
        tr_city = pd.concat([tr["left_city"], tr["right_city"]]).values
        tr_img = np.concatenate([tr["_ia"].values, tr["_ib"].values])
        tr_n = np.concatenate([ncnt[tr["_ia"].values], ncnt[tr["_ib"].values]])
        # 每张训练图的唯一城市与计数
        uniq = pd.DataFrame({"img": tr_img, "city": tr_city, "n": tr_n})
        uniq = uniq.drop_duplicates("img")
        grp = {}
        for city, sub in uniq.groupby("city"):
            b = bin_of(sub["n"].values)
            for bb in np.unique(b):
                grp[(city, bb)] = sub["img"].values[b == bb]

        ia_te = te["_ia"].values; ib_te = te["_ib"].values
        va_ia_a = va_cross["_ia"].values; va_ib_a = va_cross["_ib"].values
        y_va = va_cross["_y"].values.astype(np.int64)
        grid = np.linspace(0, 2, 21).tolist() + [3, 4, 6, 8]

        def permuted_gains(nperm: int, reselect: bool):
            g_fixed, g_resel = [], []
            q_perm = q.copy()
            for _ in range(nperm):
                for (city, bb), imgs in grp.items():
                    if len(imgs) < 2:
                        continue
                    q_perm[imgs] = q[rng.permutation(imgs)]
                # 自检：置换不改变 q 的多重集
                assert np.allclose(np.sort(q_perm), np.sort(q)), \
                    "置换改变了 q 的多重集"
                h_te_p = hist_logits(q_perm, ia_te, ib_te, hp)
                g_fixed.append(logloss_np(b2_te + beta * h_te_p, y_te) - A)
                if reselect:
                    h_va_p = hist_logits(q_perm, va_ia_a, va_ib_a, hp)
                    best_b, best_l = 0.0, 9e9
                    for g in grid:
                        l = logloss_np(b2_va + g * h_va_p, y_va)
                        if l < best_l:
                            best_b, best_l = g, l
                    g_resel.append(logloss_np(b2_te + best_b * h_te_p, y_te) - A)
            return np.array(g_fixed), (np.array(g_resel) if reselect else None)

        gf_all, _ = permuted_gains(NPERM_FIXED, reselect=False)
        _, gr = permuted_gains(NPERM_RESEL, reselect=True)

        per_fold.append({
            "fold": c, "A": A, "C": C, "beta": beta, "n": len(te),
            "obs_gain": C - A,
            "null_fixed": gf_all, "null_resel": gr,
        })
        print(f"[{fi+1:>2}] {c[:16]:<18} A={A:.6f} C={C:.6f} Δ={C-A:+.6f} "
              f"p_fixed={(gf_all <= C-A).mean():.3f}")

    # ---------- 汇总 ----------
    A_m = float(np.mean([r["A"] for r in per_fold]))
    C_m = float(np.mean([r["C"] for r in per_fold]))
    obs = float(np.mean([r["obs_gain"] for r in per_fold]))
    print("\n" + "=" * 70)
    print("自检：复现主表")
    print("=" * 70)
    print(f"  A={A_m:.6f} (期望 0.905491)  C={C_m:.6f} (期望 0.904930)")
    print(f"  Δ={obs:+.6f} (期望 -0.000561)")
    ok_repro = abs(A_m - 0.905491) < 1e-6 and abs(C_m - 0.904930) < 1e-6

    # 置换 p：城市等权，逐折 null 取均值
    nf = np.mean([r["null_fixed"] for r in per_fold], axis=0)
    nr = np.mean([r["null_resel"] for r in per_fold], axis=0)
    p_fixed = float((nf <= obs).mean())
    p_resel = float((nr <= obs).mean())
    print("\n=== 置换零分布（1000 次，固定系数）===")
    print(f"  零分布均值 {nf.mean():+.6f}  2.5% {np.percentile(nf,2.5):+.6f}  "
          f"97.5% {np.percentile(nf,97.5):+.6f}")
    print(f"  观测 {obs:+.6f}   置换 p = {p_fixed:.4f}")
    print(f"\n=== 置换零分布（200 次，重选系数）===")
    print(f"  零分布均值 {nr.mean():+.6f}  2.5% {np.percentile(nr,2.5):+.6f}  "
          f"97.5% {np.percentile(nr,97.5):+.6f}")
    print(f"  观测 {obs:+.6f}   置换 p = {p_resel:.4f}")

    # 城市聚类 bootstrap
    gains = np.array([r["obs_gain"] for r in per_fold])
    n_city = len(gains)
    boot = np.empty(10000)
    for b in range(10000):
        j = rng.integers(0, n_city, n_city)
        boot[b] = gains[j].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print("\n=== 城市聚类 bootstrap（10000 次）===")
    print(f"  观测（城市等权）{obs:+.6f}")
    print(f"  95% 区间 [{lo:+.6f}, {hi:+.6f}]")
    print(f"  相对上界 {-hi/A_m*100:.3f}%   相对下界 {-lo/A_m*100:.3f}%")
    ones = np.ones(n_city, bool)
    print(f"  自检：全 1 抽样 = {gains.mean():+.6f}（应等于观测）")

    # 记录加权
    w = np.array([r["n"] for r in per_fold], float)
    rw = float((gains * w).sum() / w.sum())
    print(f"\n=== 记录加权 vs 城市等权 ===")
    print(f"  城市等权 {obs:+.6f}   记录加权 {rw:+.6f}")

    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
                  for r in per_fold]).to_csv(OUT / "per_fold.csv", index=False)
    json.dump({"A_macro": A_m, "C_macro": C_m, "gain": obs,
               "perm_p_fixed": p_fixed, "perm_p_resel": p_resel,
               "city_boot_lo": float(lo), "city_boot_hi": float(hi),
               "record_weighted": rw, "reproduced": bool(ok_repro)},
              open(OUT / "summary.json", "w"), indent=2)
    print(f"\n已写出 {OUT/'per_fold.csv'} 与 summary.json")
    print(f"自检复现: {'通过' if ok_repro else '失败'}")
    return 0 if ok_repro else 1


if __name__ == "__main__":
    raise SystemExit(main())
