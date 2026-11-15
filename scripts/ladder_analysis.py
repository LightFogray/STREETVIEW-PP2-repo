"""编码器阶梯分析：历史增益是否随视觉模型质量单调下降。

输入
----
outputs/converge/diff_e60_s0_<backbone>.csv
每折一行，含 A（纯视觉损失）与 C（加历史后损失）

主张
----
若 C-A 的相对增益随 A 下降（模型变强）而单调收缩，则
「众包历史的边际价值是视觉模型质量的递减函数」成立，
论文的上界就不是单点数字，而是一条关系。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\converge")
RES = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\ladder")

ORDER = [("resnet18", "ResNet-18", 512),
         ("convnext_tiny", "ConvNeXt-Tiny", 768),
         ("clip_vit_l14", "CLIP ViT-L/14", 768)]


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, label, dim in ORDER:
        p = OUT / f"diff_e60_s0_{key}.csv"
        if not p.exists():
            print(f"  跳过 {key}（缺 {p.name}）")
            continue
        d = pd.read_csv(p)
        a, c = d["A"].mean(), d["C"].mean()
        delta = (d["C"] - d["A"]).mean()
        rows.append(dict(backbone=key, label=label, dim=dim,
                         A=a, C=c, delta=delta,
                         rel_pct=-delta / a * 100,
                         folds_better=int((d["C"] < d["A"]).sum()),
                         n_folds=len(d)))
        print(f"  {label:<16} dim={dim:<4} A={a:.6f}  Δ={delta:+.6f}  "
              f"相对 {rows[-1]['rel_pct']:.3f}%  {rows[-1]['folds_better']}/{len(d)} 折")

    if len(rows) < 2:
        print("  可用档位不足 2 个，无法做趋势检验")
        return 1

    df = pd.DataFrame(rows)
    df.to_csv(RES / "ladder.csv", index=False)

    print("\n=== 趋势检验 ===")
    if len(df) >= 3:
        rho, p = stats.spearmanr(df["A"], df["rel_pct"])
        print(f"  Spearman(A, 相对增益) = {rho:+.3f}  p = {p:.4f}  (n={len(df)})")
        print(f"  单调下降: {bool((df.sort_values('A')['rel_pct'].diff().dropna() < 0).all())}")
        # 增益比值：最弱 vs 最强
        d1, d2 = df["delta"].iloc[np.argmax(df["A"])], df["delta"].iloc[np.argmin(df["A"])]
        print(f"  最弱档增益 / 最强档增益 = {d1/d2:.2f}×")
    else:
        print(f"  仅 {len(df)} 档，报告方向不报 p 值")

    with open(RES / "ladder.json", "w") as fp:
        json.dump({"rows": rows}, fp, indent=2, default=float)
    print(f"\n已写入 {RES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
