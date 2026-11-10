"""用对手强度加权的历史分数（Davidson/Bradley-Terry）替换净胜分。

要回答的问题
------------
现有的历史摘要 q = (W-L)/(n+8) 完全忽略"赢了谁"。
审稿人最可能的质疑是：不是历史没用，是这个摘要太粗糙。

做法
----
每个折只用该折训练票，拟合带平局的 Bradley-Terry（Davidson）模型：

    P(left)  ∝ exp(θ_a)
    P(right) ∝ exp(θ_b)
    P(tie)   ∝ exp(ν + (θ_a+θ_b)/2)

θ 上加 L2 收缩，使票数少的图向 0 收缩（等价于原摘要的 shrinkage）。
然后把这个 θ 当作新的历史分数，重跑同一套 A/C 对照。

三组
----
A       仅视觉（B2）
C_net   视觉 + 净胜分历史（原方案）
C_bt    视觉 + Davidson 强度历史（本次）

用法：
    python bt_history.py --selfcheck
    python bt_history.py --folds 15
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
from anchor_history_fusion import (DiffMLP, fit_beta, fit_hist_model,  # noqa: E402
                                   hist_logits, history_quality, logloss)

DATA = Path(r"<PROJECT>/STREETVIEW-PP2/data")
LABELS = {"left": 0, "equal": 1, "right": 2}


def fit_davidson(theta: torch.Tensor, ia: torch.Tensor, ib: torch.Tensor,
                 y: torch.Tensor, nu: torch.Tensor, epochs: int = 30,
                 l2: float = 1e-4, bs: int = 8192, lr: float = 0.05) -> tuple:
    """在训练票上拟合带平局的 Bradley-Terry 强度。"""
    opt = torch.optim.Adam([theta, nu], lr=lr)
    n = len(ia)
    for ep in range(epochs):
        perm = torch.randperm(n, device=ia.device)
        for k in range(0, n, bs):
            j = perm[k:k + bs]
            ta, tb = theta[ia[j]], theta[ib[j]]
            logits = torch.stack([ta, 0.5 * (ta + tb) + nu, tb], dim=-1)
            loss = F.cross_entropy(logits, y[j]) + l2 * (theta ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return theta, nu


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--folds", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=4, help="B2 的 epoch")
    ap.add_argument("--bt-epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=DATA.parent / "outputs" / "bt_history")
    args = ap.parse_args()

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
    print(f"device={device}  可用票 {len(df):,}")

    def masks(f):
        c, v = f["test_city"], f["val_city"]
        in_t = (df["left_city"] == c) | (df["right_city"] == c)
        in_v = (df["left_city"] == v) | (df["right_city"] == v)
        return (df[(~in_t) & (~in_v)],
                df[in_v & df["same_city"] & (~in_t)],
                df[(in_v & (~in_t)) & (~df["same_city"])],
                df[in_t & (~df["same_city"])], c, v)

    if args.selfcheck:
        tr, va_same, va_cross, te, c, v = masks(folds[0])
        print(f"  折1 {c} val={v}  train={len(tr):,} test_cross={len(te):,}")
        ia = torch.tensor(tr["_ia"].values, device=device)
        ib = torch.tensor(tr["_ib"].values, device=device)
        yy = torch.tensor(tr["_y"].values, dtype=torch.long, device=device)
        theta = torch.zeros(n_img, device=device, requires_grad=True)
        nu = torch.tensor(0.0, device=device, requires_grad=True)
        t0 = time.time()
        th, nuv = fit_davidson(theta, ia, ib, yy, nu, epochs=3)
        print(f"  3 epoch 用时 {time.time()-t0:.1f}s   nu={float(nuv):+.3f}")
        with torch.no_grad():
            vals = th.detach()
        active = np.zeros(n_img, bool); active[tr["_ia"].values] = True; active[tr["_ib"].values] = True
        print(f"  活跃图像 {active.sum():,}   非零 θ {int((vals.abs()>1e-6).sum()):,}")
        print(f"  θ 范围 [{float(vals.min()):.3f}, {float(vals.max()):.3f}]  "
              f"std {float(vals[active].std()):.3f}")
        # 与净胜分的相关性
        q, n = history_quality(tr, n_img)
        m = active
        r = np.corrcoef(vals.cpu().numpy()[m], q[m])[0, 1]
        print(f"  θ 与净胜分 q 的相关性 r = {r:+.3f}")
        ok = (float(vals.std()) > 1e-3) and abs(r) > 0.3
        print(f"  SELFCHECK {'OK' if ok else 'FAILED'}")
        return 0 if ok else 1

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for fi, f in enumerate(folds[: args.folds]):
        tr, va_same, va_cross, te, c, v = masks(f)
        if len(va_cross) < 100 or len(te) < 100:
            continue

        # ---------- 1. 训练 B2 ----------
        torch.manual_seed(args.seed)
        model = DiffMLP().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        ia = torch.tensor(tr["_ia"].values, device=device)
        ib = torch.tensor(tr["_ib"].values, device=device)
        yy = torch.tensor(tr["_y"].values, dtype=torch.long, device=device)
        best = {"ce": 9e9, "state": None}
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(len(ia), device=device)
            for k in range(0, len(ia), 8192):
                j = perm[k:k + 8192]
                loss = F.cross_entropy(model(X[ia[j]], X[ib[j]]), yy[j])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                l = model(X[torch.tensor(va_same["_ia"].values, device=device)],
                          X[torch.tensor(va_same["_ib"].values, device=device)])
                ce = float(F.cross_entropy(l, torch.tensor(va_same["_y"].values,
                                                           dtype=torch.long, device=device)))
            if ce < best["ce"]:
                best = {"ce": ce, "state": {k: t.detach().clone()
                                            for k, t in model.state_dict().items()}}
        if best["state"]:
            model.load_state_dict(best["state"])
        model.eval()
        with torch.no_grad():
            def pred(d):
                return model(X[torch.tensor(d["_ia"].values, device=device)],
                             X[torch.tensor(d["_ib"].values, device=device)]).cpu().numpy()
            b2_va, b2_te = pred(va_cross), pred(te)

        # ---------- 2. 两种历史分数 ----------
        q, _ = history_quality(tr, n_img)
        theta = torch.zeros(n_img, device=device, requires_grad=True)
        nu = torch.tensor(0.0, device=device, requires_grad=True)
        th, _ = fit_davidson(theta, ia, ib, yy, nu, epochs=args.bt_epochs)
        with torch.no_grad():
            bt = th.detach().cpu().numpy()

        y_te = te["_y"].values
        res = {"fold": c, "n_test": len(te), "seed": args.seed,
               "A_b2": logloss(b2_te, y_te)}
        for name, score in (("net", q), ("bt", bt)):
            hp = fit_hist_model(score[va_cross["_ia"].values] - score[va_cross["_ib"].values],
                                va_cross["_y"].values)
            h_va = hist_logits(score, va_cross["_ia"].values, va_cross["_ib"].values, hp)
            h_te = hist_logits(score, te["_ia"].values, te["_ib"].values, hp)
            beta = fit_beta(b2_va, h_va, va_cross["_y"].values)
            res[f"C_{name}"] = logloss(b2_te + beta * h_te, y_te)
            res[f"beta_{name}"] = beta
        rows.append(res)
        print(f"[{fi+1}] {c[:16]:<18} A={res['A_b2']:.4f}  "
              f"净胜分={res['C_net']:.4f} (β={res['beta_net']:.2f})  "
              f"BT={res['C_bt']:.4f} (β={res['beta_bt']:.2f})")

    out = pd.DataFrame(rows)
    out.to_csv(args.out / f"per_fold_seed{args.seed}.csv", index=False)
    print(f"\n用时 {(time.time()-t0)/60:.1f} 分钟  折数 {len(out)}")
    print("\n=== 均值（逐城等权）===")
    for k in ("A_b2", "C_net", "C_bt"):
        print(f"  {k:<8}{out[k].mean():.6f}")
    for k, lab in (("C_net", "净胜分历史"), ("C_bt", "BT 强度历史")):
        d = out[k] - out["A_b2"]
        rel = -d.mean() / out["A_b2"].mean() * 100
        print(f"  {lab}: Δ={d.mean():+.6f}  相对 {rel:.3f}%  更好 {int((d<0).sum())}/{len(d)} 折")
    d = out["C_bt"] - out["C_net"]
    print(f"  BT vs 净胜分: Δ={d.mean():+.6f}  更好 {int((d<0).sum())}/{len(d)} 折")
    print(f"\n已写入 {args.out/'per_fold_seed'}{args.seed}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
