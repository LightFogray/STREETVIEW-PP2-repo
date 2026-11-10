"""参考街景历史校正：四组对照。

问题
----
跨城比较里，一端常常是"训练期已被评价过的参考图"，另一端是来自未见城市的新图。
B2 只看图像外观，不知道人们过去怎样评价那张参考图。把这些历史用上，能否改善
跨城三分类 log loss？

四组
----
A  B2                          图像差向量 MLP（基线）
B  历史单独                     只用参考图在训练票里的胜负平
C  B2 + 真实历史                在 B2 的 logits 上加 β·历史 logits
D  B2 + 打乱历史                同一城市、同一票数档内打乱 q 值（负对照）

协议（对齐既有筛查）
------------------
- 15 折留一城市；训练时移除涉及测试城/验证城的票
- 4 epoch、seed 0、AdamW lr 1e-3、batch 8192
- epoch 由验证城同城票的三分类 CE 选择
- 融合系数 β 与历史模型参数全部只在验证城（跨城对）上拟合
- 无历史的一端：q 向 0 收缩（先验收缩强度 s=8 票）

用法：
    python anchor_history_fusion.py --selfcheck
    python anchor_history_fusion.py --folds 15
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


class DiffMLP(nn.Module):
    def __init__(self, dim=768, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 3))

    def forward(self, fa, fb):
        return self.net(fa - fb)


def history_quality(sub: pd.DataFrame, n_img: int) -> tuple[np.ndarray, np.ndarray]:
    """从训练票算每张图的 q = (w - l) / (n + SHRINK)，返回 (q, n)。

    向量化实现：43 万行 × 15 折，用 iterrows 会慢几十倍。
    """
    a = sub["_ia"].to_numpy(np.int64)
    b = sub["_ib"].to_numpy(np.int64)
    c = sub["choice"].to_numpy()
    left = c == "left"
    right = c == "right"
    tie = c == "equal"

    n = np.bincount(a, minlength=n_img) + np.bincount(b, minlength=n_img)
    w = (np.bincount(a[left], minlength=n_img)
         + np.bincount(b[right], minlength=n_img))
    l = (np.bincount(b[left], minlength=n_img)
         + np.bincount(a[right], minlength=n_img))
    # 平局两边各记 0.5 胜 0.5 负
    w = w + 0.5 * (np.bincount(a[tie], minlength=n_img)
                   + np.bincount(b[tie], minlength=n_img))
    l = l + 0.5 * (np.bincount(a[tie], minlength=n_img)
                   + np.bincount(b[tie], minlength=n_img))
    q = (w - l) / (n + SHRINK)
    return q.astype(np.float32), n.astype(np.float32)


def fit_hist_model(d: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    """在 [1, d] 上拟合 3 类线性模型，返回 6 个参数（3 类 × 2 特征）用 LBFGS。"""
    Xd = torch.tensor(np.stack([np.ones_like(d), d], 1), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    p = torch.zeros(2, 3, requires_grad=True)
    opt = torch.optim.LBFGS([p], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xd @ p, yt) + l2 * (p ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return p.detach().numpy()


def hist_logits(q: np.ndarray, ia: np.ndarray, ib: np.ndarray, params: np.ndarray) -> np.ndarray:
    d = q[ia] - q[ib]
    Xd = np.stack([np.ones_like(d), d], 1).astype(np.float32)
    return Xd @ params


def fit_beta(b2, hist, y, grid=None) -> float:
    """在验证集上找 β，使 CE(softmax(b2 + β·hist)) 最小。"""
    if grid is None:
        grid = np.concatenate([np.linspace(0, 2, 41), [3, 4, 6, 8]])
    best, bestb = float("inf"), 0.0
    yt = torch.tensor(y, dtype=torch.long)
    for b in grid:
        l = F.cross_entropy(torch.tensor(b2 + b * hist, dtype=torch.float32), yt)
        if float(l) < best:
            best, bestb = float(l), float(b)
    return bestb


def logloss(logits: np.ndarray, y: np.ndarray) -> float:
    p = torch.softmax(torch.tensor(logits, dtype=torch.float32), -1)
    return float(F.nll_loss(torch.log(p.clamp_min(1e-12)),
                            torch.tensor(y, dtype=torch.long)))


def stratified_shuffle(tr, q, n, seed=0):
    """Permute all training endpoints within city and training-degree bands."""
    ends = pd.concat([tr[["_ia","left_city"]].set_axis(["i","city"],axis=1),
                      tr[["_ib","right_city"]].set_axis(["i","city"],axis=1)]).drop_duplicates()
    if ends["i"].duplicated().any() or set(ends["i"]) != set(np.flatnonzero(n)):
        raise ValueError("Training endpoint cities are inconsistent.")
    ends["band"] = np.digitize(n[ends["i"]],[2,5,10,20])
    out = q.copy()
    rng = np.random.default_rng(seed)
    for _, group in ends.groupby(["city","band"],sort=True):
        ids = np.sort(group["i"].to_numpy())
        out[ids] = q[rng.permutation(ids)]
    return out


def vote_loss(z, y):
    return np.logaddexp.reduce(z.astype(np.float64),axis=1)-z[np.arange(len(y)),y]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--folds", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=DATA.parent / "outputs" / "anchor_control_audit")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
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
    print(f"device={device}  可用票 {len(df):,}  图像 {n_img:,}")

    if args.selfcheck:
        f = folds[0]
        c, v = f["test_city"], f["val_city"]
        in_t = (df["left_city"] == c) | (df["right_city"] == c)
        in_v = (df["left_city"] == v) | (df["right_city"] == v)
        tr = df[(~in_t) & (~in_v)]
        te = df[in_t & (~df["same_city"])]
        q, n = history_quality(tr, n_img)
        warm = (n[te["_ia"].values] > 0) | (n[te["_ib"].values] > 0)
        print(f"  折1 {c}  train={len(tr):,}  test_cross={len(te):,}")
        print(f"  有历史的跨城测试票 {warm.mean()*100:.1f}%  "
              f"至少一端 >=5 票 {(((n[te['_ia'].values]>=5)|(n[te['_ib'].values]>=5)).mean()*100):.1f}%")
        print(f"  q 范围 [{q.min():.3f},{q.max():.3f}]  非零 {int((n>0).sum()):,}")
        assert 0.5 < warm.mean() < 1.0, "有历史比例异常"
        print("  SELFCHECK OK")
        return 0

    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    cluster_scores = []
    t_all = time.time()
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

        # ---- 训练 B2 ----
        torch.manual_seed(0)
        model = DiffMLP().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        ia = torch.tensor(tr["_ia"].values, dtype=torch.long, device=device)
        ib = torch.tensor(tr["_ib"].values, dtype=torch.long, device=device)
        yy = torch.tensor(tr["_y"].values, dtype=torch.long, device=device)
        best = {"ce": 9e9, "state": None}
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(len(ia), device=device)
            for k in range(0, len(ia), 8192):
                j = perm[k:k + 8192]
                out = model(X[ia[j]], X[ib[j]])
                loss = F.cross_entropy(out, yy[j])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                va_l = model(X[torch.tensor(va_same["_ia"].values, device=device)],
                             X[torch.tensor(va_same["_ib"].values, device=device)])
                ce = float(F.cross_entropy(va_l, torch.tensor(va_same["_y"].values,
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
            b2_va = pred(va_cross)
            b2_te = pred(te)

        # ---- 历史模型（只在验证集跨城对上拟合）----
        hp = fit_hist_model(q[va_cross["_ia"].values] - q[va_cross["_ib"].values],
                            va_cross["_y"].values)
        h_va = hist_logits(q, va_cross["_ia"].values, va_cross["_ib"].values, hp)
        h_te = hist_logits(q, te["_ia"].values, te["_ib"].values, hp)
        beta = fit_beta(b2_va, h_va, va_cross["_y"].values)

        # ---- 打乱历史：同城同票数档内置换 q ----
        q_sh = stratified_shuffle(tr, q, n)
        h_va_sh = hist_logits(q_sh, va_cross["_ia"].values, va_cross["_ib"].values, hp)
        h_te_sh = hist_logits(q_sh, te["_ia"].values, te["_ib"].values, hp)
        beta_sh = fit_beta(b2_va, h_va_sh, va_cross["_y"].values)

        y_te = te["_y"].values
        res = {
            "fold": c, "n_test": len(te), "beta": beta, "beta_shuffled": beta_sh,
            "A_b2": logloss(b2_te, y_te),
            "B_hist_only": logloss(h_te, y_te),
            "C_b2_plus_hist": logloss(b2_te + beta * h_te, y_te),
            "D_b2_plus_shuffled": logloss(b2_te + beta_sh * h_te_sh, y_te),
            "D_fixed_beta": logloss(b2_te + beta * h_te_sh, y_te),
            "intercept_only_fixed_beta": logloss(b2_te + beta * hp[0], y_te),
        }
        # Conditional uncertainty across voters for the macro-average of fixed-fold
        # differences. Shared comparisons and voters retain their fold memberships.
        vc = te["voter_uniqueid"]
        if vc.isna().any():
            raise ValueError("Missing voter identifier; specify the cluster handling.")
        ref_loss = vote_loss(b2_te, y_te)
        true_loss = vote_loss(b2_te + beta*h_te, y_te)
        deltas = {"C_minus_A":true_loss-ref_loss,
                  "C_minus_D_fixed":true_loss-vote_loss(b2_te+beta*h_te_sh,y_te)}
        part = pd.DataFrame({k:(a-a.mean())/len(te) for k,a in deltas.items()},index=vc.to_numpy())
        cluster_scores.append(part.groupby(level=0).sum())
        warm = (n[te["_ia"].values] > 0) | (n[te["_ib"].values] > 0)
        res["warm_frac"] = float(warm.mean())
        rows.append(res)
        print(f"[{fi+1}] {c[:16]:<18} β={beta:.2f}  A={res['A_b2']:.4f} "
              f"B={res['B_hist_only']:.4f} C={res['C_b2_plus_hist']:.4f} "
              f"D={res['D_b2_plus_shuffled']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out / "per_fold.csv", index=False)
    influence = pd.concat(cluster_scores).groupby(level=0).sum() / len(out)
    se = np.sqrt((influence**2).sum()*len(influence)/(len(influence)-1))
    center = {"C_minus_A":out.C_b2_plus_hist.mean()-out.A_b2.mean(),
              "C_minus_D_fixed":out.C_b2_plus_hist.mean()-out.D_fixed_beta.mean()}
    ci = {k:{"mean":v,"se":float(se[k]),"conditional_95pct":[v-1.96*se[k],v+1.96*se[k]]}
          for k,v in center.items()}
    (args.out/"voter_conditional_ci.json").write_text(json.dumps(
        {"clusters":len(influence),"intervals":ci},indent=2),encoding="utf-8")
    print(f"\n合计用时 {(time.time()-t_all)/60:.1f} 分钟")
    print("\n=== 均值（逐城等权）===")
    for k in ("A_b2", "B_hist_only", "C_b2_plus_hist", "D_b2_plus_shuffled"):
        print(f"  {k:<20}{out[k].mean():.4f} ± {out[k].std():.4f}")
    for a, b in (("C_b2_plus_hist", "A_b2"), ("C_b2_plus_hist", "D_b2_plus_shuffled")):
        d = out[a] - out[b]
        print(f"  {a} - {b}: {d.mean():+.4f}  更好 {int((d<0).sum())}/{len(d)} 折")
    print(f"\n已写入 {args.out/'per_fold.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
