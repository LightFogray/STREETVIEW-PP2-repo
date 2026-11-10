"""共享图像评分模型（Davidson 平局模型）下的历史增益复核。

为什么用 Davidson
----------------
B2 直接吃 (f_a - f_b)，把"两张图各自有多好"和"差多少"混在一个 MLP 里。
共享评分模型换成每个图一个标量分数 s(x)，再用 Davidson 模型生成三类概率：

    P(left) ∝ exp(s_a)
    P(right) ∝ exp(s_b)
    P(tie)   ∝ nu * sqrt(exp(s_a) * exp(s_b))
    => logits = [s_a, s_b, 0.5*(s_a+s_b) + c]

这是带平局的成对比较标准模型，参数量比 B2 少得多，且交换左右自动对应对称，
回答的问题是：**历史增益只有 0.06%，是不是 B2 这个差异头的特例？**

用法：
    python shared_scorer_check.py --selfcheck
    python shared_scorer_check.py --folds 15
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

DATA = Path(r"<PROJECT>/STREETVIEW-PP2/data")
LABELS = {"left": 0, "equal": 1, "right": 2}
SHRINK = 8.0


# ---- 复用 anchor 脚本里的历史工具 ----
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_history_fusion import (fit_beta, fit_hist_model, hist_logits,  # noqa: E402
                                   history_quality, logloss)


class SharedScorer(nn.Module):
    """每图一个标量分数 + 一个平局倾向参数（Davidson）。"""

    def __init__(self, dim=768, hidden=256, dropout=0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
        self.tie = nn.Parameter(torch.tensor(0.0))

    def forward(self, fa, fb):
        sa = self.score(fa).squeeze(-1)
        sb = self.score(fb).squeeze(-1)
        # 类别顺序必须与 LABELS 一致：[left, equal, right]
        # 曾经写成 [sa, sb, tie]，等于把右图分数当平局、平局当右图（自检抓到）
        return torch.stack([sa, 0.5 * (sa + sb) + self.tie, sb], dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--folds", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--out", type=Path,
                    default=DATA.parent / "outputs" / "shared_scorer")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(DATA / "pairs.parquet")
    folds = json.loads((DATA / "splits" / "logo_folds.json").read_text(encoding="utf-8"))
    z = np.load(DATA / "features_convnext_tiny.npz")
    ids = z["ids"].tolist()
    idx = {k: i for i, k in enumerate(ids)}
    X = torch.from_numpy(z["features"].astype(np.float32)).to(device)
    n_img = X.shape[0]

    df = df[df["left"].isin(idx) & df["right"].isin(idx)].reset_index(drop=True)
    df["_ia"] = df["left"].map(idx).astype(np.int32)
    df["_ib"] = df["right"].map(idx).astype(np.int32)
    df["_y"] = df["choice"].map(LABELS).astype(np.int8)
    print(f"device={device}  可用票 {len(df):,}")

    if args.selfcheck:
        f = folds[0]
        c, v = f["test_city"], f["val_city"]
        in_t = (df["left_city"] == c) | (df["right_city"] == c)
        in_v = (df["left_city"] == v) | (df["right_city"] == v)
        tr = df[(~in_t) & (~in_v)]
        m = SharedScorer().to(device)
        ia = torch.tensor(tr["_ia"].values[:8192], device=device)
        ib = torch.tensor(tr["_ib"].values[:8192], device=device)
        y = torch.tensor(tr["_y"].values[:8192], dtype=torch.long, device=device)
        m.train()
        logits = m(X[ia], X[ib])
        loss = F.cross_entropy(logits, y)
        loss.backward()
        n_par = sum(p.numel() for p in m.parameters())
        print(f"  logits {tuple(logits.shape)}  参数量 {n_par:,}")
        print(f"  tie 参数梯度 {float(m.tie.grad):.4f}")
        # 对称性必须在 eval 模式下检查：train 模式下 dropout 会让两次前向不同
        m.eval()
        with torch.no_grad():
            l1 = m(X[ia], X[ib])
            l2 = m(X[ib], X[ia])
        swap_ok = (torch.allclose(l2[:, 0], l1[:, 2], atol=1e-5)
                   and torch.allclose(l2[:, 2], l1[:, 0], atol=1e-5)
                   and torch.allclose(l2[:, 1], l1[:, 1], atol=1e-5))
        print(f"  交换左右对称（eval 模式）: {swap_ok}")
        print(f"  SELFCHECK {'OK' if swap_ok and float(loss) == float(loss) else 'FAILED'}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for fi, f in enumerate(folds[: args.folds]):
        c, v = f["test_city"], f["val_city"]
        in_t = (df["left_city"] == c) | (df["right_city"] == c)
        in_v = (df["left_city"] == v) | (df["right_city"] == v)
        tr = df[(~in_t) & (~in_v)]
        va_same = df[in_v & df["same_city"] & (~in_t)]
        va_cross = df[(in_v & (~in_t)) & (~df["same_city"])]
        te = df[in_t & (~df["same_city"])]
        if len(va_cross) < 100 or len(te) < 100:
            continue
        q, n = history_quality(tr, n_img)

        torch.manual_seed(0)
        model = SharedScorer().to(device)
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
                best = {"ce": ce, "state": {k: v_.detach().clone()
                                            for k, v_ in model.state_dict().items()}}
        if best["state"]:
            model.load_state_dict(best["state"])
        model.eval()
        with torch.no_grad():
            def pred(d):
                return model(X[torch.tensor(d["_ia"].values, device=device)],
                             X[torch.tensor(d["_ib"].values, device=device)]).cpu().numpy()
            s_va, s_te = pred(va_cross), pred(te)

        hp = fit_hist_model(q[va_cross["_ia"].values] - q[va_cross["_ib"].values],
                            va_cross["_y"].values)
        h_va = hist_logits(q, va_cross["_ia"].values, va_cross["_ib"].values, hp)
        h_te = hist_logits(q, te["_ia"].values, te["_ib"].values, hp)
        beta = fit_beta(s_va, h_va, va_cross["_y"].values)

        y_te = te["_y"].values
        rows.append({
            "fold": c, "n_test": len(te), "beta": beta,
            "A_shared": logloss(s_te, y_te),
            "C_shared_plus_hist": logloss(s_te + beta * h_te, y_te),
        })
        print(f"[{fi+1}] {c[:16]:<18} β={beta:.2f}  A={rows[-1]['A_shared']:.4f} "
              f"C={rows[-1]['C_shared_plus_hist']:.4f}  "
              f"Δ={rows[-1]['C_shared_plus_hist']-rows[-1]['A_shared']:+.6f}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out / "per_fold.csv", index=False)
    print(f"\n用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"  A 共享评分        {out['A_shared'].mean():.6f}")
    print(f"  C 共享评分+历史    {out['C_shared_plus_hist'].mean():.6f}")
    d = out["C_shared_plus_hist"] - out["A_shared"]
    print(f"  Δ = {d.mean():+.6f}  更好 {int((d<0).sum())}/{len(d)} 折  "
          f"相对 {(out['A_shared'].mean()-out['C_shared_plus_hist'].mean())/out['A_shared'].mean()*100:.3f}%")
    print(f"\n已写入 {args.out/'per_fold.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
