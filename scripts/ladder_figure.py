"""编码器阶梯出图（修订版）。

面板 (a) 改为折级散点：每个点是一折在某一编码器下的
（视觉损失 A，相对历史增益），共 15×3=45 点，再叠三档均值。
这样能看到关系不是均值造出来的。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

CONV = r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\converge"
RES = r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\ladder"
FIG = r"D:\mycode\crime-temporal-spatio\paper_figures\revision"

ORDER = [("resnet18", "ResNet-18", 512),
         ("convnext_tiny", "ConvNeXt-Tiny", 768),
         ("clip_vit_l14", "CLIP ViT-L/14", 768)]
COLORS = ["#c0392b", "#7f8c8d", "#1b6ca8"]


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 160, "savefig.dpi": 400, "savefig.bbox": "tight",
    })

    dfs, rows = {}, []
    for key, label, dim in ORDER:
        d = pd.read_csv(f"{CONV}/diff_e60_s0_{key}.csv")
        dfs[key] = d
        a, c = d["A"].mean(), d["C"].mean()
        delta = (d["C"] - d["A"]).mean()
        better = int((d["C"] < d["A"]).sum())
        rows.append(dict(backbone=key, label=label, dim=dim, A=a, C=c,
                         delta=delta, rel_pct=-delta / a * 100,
                         folds_better=better, n_folds=len(d),
                         binom_p=float(stats.binomtest(better, len(d), 0.5,
                                                       alternative="greater").pvalue)))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"width_ratios": [1.35, 1.0],
                                          "wspace": 0.32})
    ax = axes[0]
    for i, (key, label, dim) in enumerate(ORDER):
        d = dfs[key]
        rel = -(d["C"] - d["A"]) / d["A"] * 100
        ax.scatter(d["A"], rel, s=16, alpha=0.5, color=COLORS[i],
                   edgecolor="none", label=f"{label} ({dim}-d)")
    ax.scatter([r["A"] for r in rows], [r["rel_pct"] for r in rows],
               s=80, marker="D", c=COLORS, edgecolor="black", lw=0.7, zorder=6)
    ax.plot([r["A"] for r in rows], [r["rel_pct"] for r in rows],
            ls="-", lw=1.0, color="#444444", zorder=5)

    OFFS = {"resnet18": (-18, 12), "convnext_tiny": (-34, 10), "clip_vit_l14": (-8, -18)}
    for r, col in zip(rows, COLORS):
        dx, dy = OFFS[r["backbone"]]
        ax.annotate(r["label"], (r["A"], r["rel_pct"]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha="center", fontsize=7.2, color=col)

    ax.axhline(0, ls="-", lw=0.8, color="#999999")
    ax.set_xlabel("Visual-model loss, city average (lower = stronger encoder)")
    ax.set_ylabel("Relative history gain (%)")
    ax.set_title("(a) Gain against encoder strength", fontsize=8.5)
    ax.invert_xaxis()
    ax.legend(frameon=False, loc="upper right", fontsize=6.8, handletextpad=0.4)

    ax = axes[1]
    xs = np.arange(len(rows))
    ax.bar(xs, [r["folds_better"] for r in rows], color=COLORS,
           edgecolor="black", lw=0.5, width=0.62)
    ax.axhline(rows[0]["n_folds"] / 2, ls="--", lw=0.9, color="#555555")
    ax.annotate("half the folds", (0.02, rows[0]["n_folds"] / 2 + 0.35),
                fontsize=6.8, ha="left", color="#555555")
    for i, r in enumerate(rows):
        ax.annotate(f"{r['folds_better']}", (i, r["folds_better"]),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7.5)
        ax.annotate(f"p={r['binom_p']:.2f}" if r["binom_p"] > 0.01 else "p<0.01",
                    (i, 0.4), ha="center", fontsize=6.5, color="white")
    ax.set_xticks(xs)
    ax.set_xticklabels(["ResNet-18", "ConvNeXt\n-Tiny", "CLIP\nViT-L/14"], fontsize=7)
    ax.set_ylabel("Folds where history helps (of 15)")
    ax.set_ylim(0, 17.5)
    ax.set_title("(b) Consistency across folds", fontsize=8.5)

    fig.savefig(f"{FIG}/fig6_encoder_ladder.pdf")
    fig.savefig(f"{FIG}/fig6_encoder_ladder.png")
    plt.close(fig)
    print("已出图 fig6_encoder_ladder")

    ratio = rows[0]["rel_pct"] / rows[-1]["rel_pct"]
    with open(f"{RES}/ladder.json", "w") as fp:
        json.dump({"rows": rows, "ratio": float(ratio),
                   "monotone": bool(np.all(np.diff([r["rel_pct"] for r in rows]) < 0))},
                  fp, indent=2, default=float)
    print(f"  最弱/最强 = {ratio:.2f}×   单调 {np.all(np.diff([r['rel_pct'] for r in rows]) < 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
