"""补做两个决定性实验，回应审阅意见。

问题
----
1. 主模型（差向量头）只训了 4 epoch，而共享评分头有收敛对照。
   我们自己证明 4 epoch 会把历史增益放大约 4.6 倍，所以主模型的 0.062%
   不能和收敛的 1.26% 直接比。必须补差向量头的收敛值。
2. 补充材料里交互头（差 + 和 + 积）在 T2 上优于差向量头（0.90433 vs 0.90560），
   但从未在它上面测过历史校正。必须补。

做法
----
对每个 (head, epochs) 组合，逐折：
    训练 -> 验证城同城票选轮 -> 训练城跨城票上算历史增益（净胜分摘要）
只算净胜分摘要，跳过 Davidson 拟合以节省时间。

用法：
    python converge_heads.py --head diff --epochs 150
    python converge_heads.py --head interact --epochs 150
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
from anchor_history_fusion import (fit_beta, fit_hist_model, hist_logits,
                                   history_quality, logloss)

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
LABELS = {"left": 0, "equal": 1, "right": 2}


class Head(nn.Module):
    """mode='diff' 只用 f_a-f_b；mode='interact' 再加上 f_a+f_b 与 f_a*f_b。"""

    def __init__(self, dim=768, hidden=256, dropout=0.1, mode="diff"):
        super().__init__()
        self.mode = mode
        in_dim = dim if mode == "diff" else dim * 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 3))

    def forward(self, fa, fb):
        if self.mode == "diff":
            z = fa - fb
        else:
            z = torch.cat([fa - fb, fa + fb, fa * fb], dim=-1)
        return self.net(z)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["diff", "interact"], default="diff")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=DATA.parent / "outputs" / "converge")
    ap.add_argument("--features", type=Path,
                    default=DATA / "features_convnext_tiny.npz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(DATA / "pairs.parquet")
    folds = json.loads((DATA / "splits" / "logo_folds.json").read_text(encoding="utf-8"))
    z = np.load(args.features)
    idx = {k: i for i, k in enumerate(z["ids"].tolist())}
    X = torch.from_numpy(z["features"].astype(np.float32)).to(device)
    n_img = X.shape[0]

    df = df[df["left"].isin(idx) & df["right"].isin(idx)].reset_index(drop=True)
    df["_ia"] = df["left"].map(idx).astype(np.int32)
    df["_ib"] = df["right"].map(idx).astype(np.int32)
    df["_y"] = df["choice"].map(LABELS).astype(np.int8)
    print(f"head={args.head} epochs={args.epochs} seed={args.seed} device={device}")

    args.out.mkdir(parents=True, exist_ok=True)
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
        model = Head(dim=int(X.shape[1]), mode=args.head).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        ia = torch.tensor(tr["_ia"].values, device=device)
        ib = torch.tensor(tr["_ib"].values, device=device)
        yy = torch.tensor(tr["_y"].values, dtype=torch.long, device=device)
        va_ia = torch.tensor(va_same["_ia"].values, device=device)
        va_ib = torch.tensor(va_same["_ib"].values, device=device)
        va_y = torch.tensor(va_same["_y"].values, dtype=torch.long, device=device)

        best = {"ce": 9e9, "ep": 0, "state": None}
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(len(ia), device=device)
            for k in range(0, len(ia), 8192):
                j = perm[k:k + 8192]
                loss = F.cross_entropy(model(X[ia[j]], X[ib[j]]), yy[j])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                l = model(X[va_ia], X[va_ib])
                ce = float(F.cross_entropy(l, va_y))
            if ce < best["ce"]:
                best = {"ce": ce, "ep": ep,
                        "state": {k: t.detach().clone()
                                  for k, t in model.state_dict().items()}}
        if best["state"]:
            model.load_state_dict(best["state"])
        model.eval()

        with torch.no_grad():
            def pred(d):
                return model(X[torch.tensor(d["_ia"].values, device=device)],
                             X[torch.tensor(d["_ib"].values, device=device)]).cpu().numpy()
            b2_va, b2_te = pred(va_cross), pred(te)

        q, _ = history_quality(tr, n_img)
        hp = fit_hist_model(q[va_cross["_ia"].values] - q[va_cross["_ib"].values],
                            va_cross["_y"].values)
        h_va = hist_logits(q, va_cross["_ia"].values, va_cross["_ib"].values, hp)
        h_te = hist_logits(q, te["_ia"].values, te["_ib"].values, hp)
        beta = fit_beta(b2_va, h_va, va_cross["_y"].values)

        y_te = te["_y"].values
        rows.append({
            "fold": c, "seed": args.seed, "head": args.head,
            "epochs": args.epochs, "best_epoch": best["ep"], "n_test": len(te),
            "A": logloss(b2_te, y_te),
            "C": logloss(b2_te + beta * h_te, y_te),
            "beta": beta,
        })
        r = rows[-1]
        print(f"[{fi+1:>2}] {c[:16]:<18} ep={best['ep']:>3} A={r['A']:.6f} "
              f"C={r['C']:.6f} Δ={r['C']-r['A']:+.6f}")

    out = pd.DataFrame(rows)
    bname = args.features.stem.replace("features_", "")
    tag = f"{args.head}_e{args.epochs}_s{args.seed}_{bname}"
    out.to_csv(args.out / f"{tag}.csv", index=False)
    d = out["C"] - out["A"]
    print(f"\n用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"  A      {out['A'].mean():.6f}")
    print(f"  C      {out['C'].mean():.6f}")
    print(f"  Δ      {d.mean():+.6f}   相对 {-d.mean()/out['A'].mean()*100:.3f}%"
          f"   更好 {int((d<0).sum())}/{len(d)} 折")
    print(f"  选中的 epoch: 中位 {out['best_epoch'].median():.0f}  范围 "
          f"[{out['best_epoch'].min()}, {out['best_epoch'].max()}]")
    print(f"已写入 {args.out/(tag+'.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
