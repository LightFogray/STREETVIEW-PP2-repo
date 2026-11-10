"""生成论文所需的三张图（第 2、4、5 张）以及骨干敏感性图（第 3 张）。

Fig 2  参考图历史可得性分布（按训练票数档）
Fig 3  骨干敏感性：视觉损失 vs 历史增益
Fig 4  研究区地图：56 城，15 个留出测试城
Fig 5  平局分析：AP 与平局率，以及 argmax 从不选中平局

全部输出 PDF（矢量）+ PNG（300 dpi）到 manuscript/figures。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA = Path(r"<PROJECT>/STREETVIEW-PP2/data")
OUT = Path(r"<PROJECT>/STREETVIEW-PP2/manuscript/figures")
GEO = DATA / "geo" / "ne_110m_admin_0_countries.geojson"
RLAB = Path(r"<PROJECT>/STREETVIEW-PP2/outputs/anchor_fusion/per_fold.csv")
AUD = Path(r"<PROJECT>/STREETVIEW-PP2/outputs/anchor_control_audit/per_fold.csv")
EQ = Path(r"<PROJECT>/STREETVIEW-PP2/outputs/equal_interaction_screen/metrics.csv")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 400, "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=400)
    plt.close(fig)
    print(f"  写出 {name}.pdf / .png")


# ---------------------------------------------------------------- Fig 2
def fig_history_availability() -> None:
    from anchor_history_fusion import history_quality

    df = pd.read_parquet(DATA / "pairs.parquet")
    folds = json.loads((DATA / "splits" / "logo_folds.json").read_text(encoding="utf-8"))
    z = np.load(DATA / "features_convnext_tiny.npz")
    idx = {k: i for i, k in enumerate(z["ids"].tolist())}
    df = df[df["left"].isin(idx) & df["right"].isin(idx)].reset_index(drop=True)
    df["_ia"] = df["left"].map(idx).astype(np.int32)
    df["_ib"] = df["right"].map(idx).astype(np.int32)
    n_img = len(idx)

    ns = []
    for f in folds:
        c, v = f["test_city"], f["val_city"]
        in_t = (df["left_city"] == c) | (df["right_city"] == c)
        in_v = (df["left_city"] == v) | (df["right_city"] == v)
        tr = df[(~in_t) & (~in_v)]
        te = df[in_t & (~df["same_city"])]
        _, n = history_quality(tr, n_img)
        # 一张图在训练中的比较次数；取两端中较大的那个
        ns.append(np.maximum(n[te["_ia"].values], n[te["_ib"].values]))
    alln = np.concatenate(ns)
    print(f"  T2 测试记录 {len(alln):,}  有历史 {(alln>0).mean()*100:.2f}%  "
          f">=5 {(alln>=5).mean()*100:.2f}%")

    edges = [0, 1, 2, 5, 10, 20, 1e9]
    labels = ["0\n(none)", "1", "2–4", "5–9", "10–19", "≥20"]
    counts = [int(((alln >= edges[i]) & (alln < edges[i + 1])).sum())
              for i in range(len(edges) - 1)]
    frac = np.array(counts) / len(alln) * 100

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    colors = ["#c9c9c9"] + ["#4C72B0"] * 5
    bars = ax.bar(range(6), frac, color=colors, edgecolor="white", linewidth=0.6)
    for i, (b, fr) in enumerate(zip(bars, frac)):
        ax.text(b.get_x() + b.get_width() / 2, fr + 0.6, f"{fr:.1f}",
                ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(6))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Training comparisons of the better-known\nreference image", fontsize=7.5)
    ax.set_ylabel("Share of T2 test records (%)")
    ax.set_ylim(0, max(frac) * 1.28)
    ax.text(0.02, 0.97,
            f"history available {100-frac[0]:.1f}%\n"
            f"at least five comparisons {100-frac[0]-frac[1]-frac[2]:.1f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            linespacing=1.35)
    save(fig, "fig3_history_availability")


# ---------------------------------------------------------------- Fig 3
def fig_backbone_sensitivity() -> None:
    """三个头 × 两个训练预算。同一头的两点连线，表示预算的作用。"""
    series = {
        "difference head": ("#4C72B0", "o",
                            [(0.905491, 0.062), (0.906153, 0.055)]),
        "difference + sum + product": ("#55A868", "s",
                                       [(0.904320, 0.067), (0.905342, 0.081)]),
        "shared scorer": ("#C44E52", "^",
                          [(0.9664, 5.83), (0.9201, 1.26)]),
    }
    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    for name, (color, marker, pts) in series.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-", color=color, lw=0.9, alpha=0.55, zorder=1)
        ax.scatter(xs, ys, s=34, color=color, marker=marker, zorder=3,
                   edgecolor="white", linewidth=0.6, label=name)
        # 在第二点（更长预算）旁标出预算
        ax.annotate("4 ep" if len(ys) == 2 else "",
                    (xs[0], ys[0]), xytext=(0, -9), textcoords="offset points",
                    fontsize=5.8, ha="center", color=color)
        ax.annotate("longer", (xs[1], ys[1]), xytext=(6, 4),
                    textcoords="offset points", fontsize=5.8, ha="left", color=color)
    ax.set_yscale("log")
    ax.set_xlabel("Log loss without history (lower is better)")
    ax.set_ylabel("History gain (%, log scale)")
    ax.set_xlim(0.900, 0.988)
    ax.set_ylim(0.03, 14)
    ax.axvspan(0.900, 0.9075, color="#eeeeee", zorder=0)
    ax.text(0.9038, 9.0, "best visual\nquality", fontsize=5.8, color="#777777",
            ha="center", va="top", linespacing=1.2)
    ax.legend(frameon=False, loc="lower left", fontsize=6.0, handletextpad=0.4)
    save(fig, "fig5_head_sensitivity")


# ---------------------------------------------------------------- Fig 4
def fig_map() -> None:
    import geopandas as gpd

    df = pd.read_parquet(DATA / "pairs.parquet")
    folds = json.loads((DATA / "splits" / "logo_folds.json").read_text(encoding="utf-8"))
    test_cities = {f["test_city"] for f in folds}

    # 每个城市：端点坐标中位数 + 总票数
    rows = []
    for side in ("left", "right"):
        rows.append(df[[f"{side}_city", f"{side}_lat", f"{side}_lon"]]
                    .rename(columns={f"{side}_city": "city",
                                     f"{side}_lat": "lat", f"{side}_lon": "lon"}))
    e = pd.concat(rows, ignore_index=True)
    g = e.groupby("city").agg(lat=("lat", "median"), lon=("lon", "median"),
                              n=("lat", "size")).reset_index()
    print(f"  城市 {len(g)}  测试城 {len(test_cities)}")

    world = gpd.read_file(GEO)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    world.plot(ax=ax, color="#efefef", edgecolor="white", linewidth=0.3)

    g["is_test"] = g["city"].isin(test_cities)
    style = {False: ("#8FA8C8", "other city"),
             True: ("#C44E52", "held-out test city")}
    for flag, (color, _lbl) in style.items():
        s = g[g["is_test"] == flag]
        ax.scatter(s["lon"], s["lat"], s=np.sqrt(s["n"]) * 1.15, c=color,
                   alpha=0.80, edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_xlim(-170, 180)
    ax.set_ylim(-58, 78)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    # 用固定尺寸的代理对象做图例：
    # 直接让 legend 接管散点会把各组平均气泡直径当成图例标记，两组因此大小不一
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=5.0,
                      markerfacecolor=color, markeredgecolor="white",
                      markeredgewidth=0.5, label=lbl)
               for color, lbl in style.values()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=2, frameon=False, handletextpad=0.5,
              columnspacing=2.4, borderaxespad=0.0)
    # 标题不进图：期刊会在正文重新排版图注，图内再放一遍会重复且无法编辑
    save(fig, "fig1_study_cities")


# ---------------------------------------------------------------- Fig 5
def fig_ties() -> None:
    d = pd.read_csv(EQ)
    d = d[d["arm"] == "diff"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))

    ax = axes[0]
    for test, color, marker, lbl in (("same", "#4C72B0", "o", "T1 within-city"),
                                     ("cross", "#DD8452", "s", "T2 cross-city")):
        s = d[d["test"] == test]
        ax.scatter(s["tie_prevalence"] * 100, s["equal_ap"], s=18, c=color,
                   marker=marker, alpha=0.85, edgecolor="white", linewidth=0.3, label=lbl)
        lo, hi = s["tie_prevalence"].min() * 100, s["tie_prevalence"].max() * 100
        # 基准：AP 的随机水平等于平局率本身（比例，不是百分数）
        ax.plot([lo, hi], [lo / 100, hi / 100], color=color, ls=":", lw=0.9)
    ax.set_xlabel("Tie prevalence (%)")
    ax.set_ylabel("Tie average precision")
    ax.legend(frameon=False, loc="upper left")
    ax.text(-0.02, 1.03, "(a)", transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="right")
    ax.annotate("dotted: random-ranking level (AP = prevalence)",
                xy=(19.6, 0.198), xytext=(14.2, 0.118), fontsize=6.3, color="#555555",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="#888888"))
    ax.set_ylim(0.10, 0.27)

    ax = axes[1]
    tot = d.groupby("test")["n"].sum()
    sel = d.groupby("test")["equal_argmax_n"].sum()
    share = (sel / tot * 100).reindex(["same", "cross"])
    prev = d.groupby("test")["tie_n"].sum() / tot * 100
    x = np.arange(2)
    ax.bar(x - 0.19, prev.reindex(["same", "cross"]).values, width=0.36,
           color="#b0b0b0", label="actual tie rate")
    ax.bar(x + 0.19, share.reindex(["same", "cross"]).values, width=0.36,
           color="#C44E52", label="ties chosen by argmax")
    ax.set_xticks(x)
    ax.set_xticklabels(["T1 within-city", "T2 cross-city"])
    ax.set_ylabel("Share of test records (%)")
    ax.legend(frameon=False)
    ax.text(-0.02, 1.03, "(b)", transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="right")
    for i, v in enumerate(share.reindex(["same", "cross"]).values):
        ax.text(i + 0.19, v + 0.4, "0.0", ha="center", fontsize=6.5)

    fig.tight_layout()
    save(fig, "fig2_tie_analysis")


def fig_reference_history_by_city() -> None:
    """逐城差值哑铃图：真实历史 vs 固定系数的匹配打乱。"""
    d = pd.read_csv(AUD)
    d["true"] = d["C_b2_plus_hist"] - d["A_b2"]
    d["shuf"] = d["D_fixed_beta"] - d["A_b2"]
    d = d.sort_values("true")

    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    y = np.arange(len(d))
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.plot([r["true"] * 1e3, r["shuf"] * 1e3], [yi, yi],
                color="#cfcfcf", lw=1.1, zorder=1, solid_capstyle="round")
    ax.scatter(d["true"] * 1e3, y, s=26, color="#1f5f7a", zorder=3,
               edgecolor="white", linewidth=0.5, label="True history")
    ax.scatter(d["shuf"] * 1e3, y, s=26, marker="s", facecolor="none",
               edgecolor="#B5651D", linewidth=0.9, zorder=3,
               label="Matched shuffle, fixed coefficient")
    ax.axvline(0, color="#888888", ls="--", lw=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(d["fold"], fontsize=6.8)
    ax.set_xlabel("Change from visual baseline in log loss (× 10$^{-3}$)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.set_ylim(-0.8, len(d) - 0.2)
    save(fig, "fig4_reference_history_by_city")


def main() -> int:
    print("Fig 1 研究区地图 ...")
    fig_map()
    print("Fig 2 平局分析 ...")
    fig_ties()
    print("Fig 3 历史可得性 ...")
    fig_history_availability()
    print("Fig 4 逐城差值 ...")
    fig_reference_history_by_city()
    print("Fig 5 骨干敏感性 ...")
    fig_backbone_sensitivity()
    print("\n全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
