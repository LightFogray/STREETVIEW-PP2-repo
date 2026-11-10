"""Day 1（续）：留一城市划分 + 泄漏断言。

划分协议
--------
对每个候选测试城 C（同城票 >= 300）：
    train     两端都不在 C 的票
    val       两端都在 V 的票（V 是另一个留出城，用于模型选择，不含 C 的任何信息）
    test_same 两端都在 C 的票        ← T1，检验"未见过城市内部的排序"
    test_cross 恰好一端在 C 的票     ← T2，检验"跨城比较中含未见城市"

关键约束
--------
1. C 的图像绝不进入 train / val
2. val 城 V 的图像也不进入 train（保持 val 与 test 同构）
3. 图对不跨集合
4. 86% 的投票者跨城 —— 这**不是**泄漏（同一投票者在不同集合投的是不同图对），
   但必须在报告里声明，且 bootstrap 要按投票者聚类

用法：
    python make_splits.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(r"<PROJECT>/STREETVIEW-PP2/data")
OUT = D / "splits"
MIN_SAME_CITY = 300


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(D / "pairs.parquet")
    print(f"载入 {len(df):,} 票")

    same = df[df["same_city"]]
    per_city = same["left_city"].value_counts()
    cand = [c for c in per_city.index if per_city[c] >= MIN_SAME_CITY]
    print(f"同城票 >= {MIN_SAME_CITY} 的候选测试城: {len(cand)}")

    summary = []
    for k, test_city in enumerate(cand):
        # 验证城：从候选里轮转取一个不同的城，且不与测试城相同
        val_city = cand[(k + 1) % len(cand)]
        if val_city == test_city:
            continue

        in_test = (df["left_city"] == test_city) | (df["right_city"] == test_city)
        in_val = (df["left_city"] == val_city) | (df["right_city"] == val_city)

        test_same = in_test & df["same_city"]
        test_cross = in_test & (~df["same_city"])
        val_mask = in_val & df["same_city"] & (~in_test)
        train = (~in_test) & (~in_val)

        fold = {
            "test_city": test_city, "val_city": val_city,
            "n_train": int(train.sum()),
            "n_val": int(val_mask.sum()),
            "n_test_same": int(test_same.sum()),
            "n_test_cross": int(test_cross.sum()),
        }

        # ---- 泄漏断言 ----
        # 注意：T2（跨城测试）按设计就是"一端在测试城、一端在训练城"，
        # 所以训练城市的图必然同时出现在 train 和 test_cross 里。这不是泄漏。
        # 真正必须成立的是下面三条。
        def imgs(mask):
            s = df[mask]
            return set(s["left"]) | set(s["right"])

        tr_i = imgs(train)
        va_i = imgs(val_mask)
        # (a) T1 的图像与 train 完全不相交
        t1_i = imgs(test_same)
        fold["assert_T1_img_disjoint"] = len(tr_i & t1_i) == 0
        # (b) val 的图像与 train 不相交
        fold["assert_val_img_disjoint"] = len(tr_i & va_i) == 0
        # (c) 测试城的图像绝不进入 train
        fold["assert_test_city_absent"] = not any(
            c == test_city for c in
            set(df.loc[train, "left_city"]) | set(df.loc[train, "right_city"]))
        # (d) 图对不跨集合
        def pairs(mask):
            s = df[mask]
            return set(zip(np.minimum(s["left"], s["right"]),
                           np.maximum(s["left"], s["right"])))
        fold["assert_pair_disjoint"] = len(pairs(train) & pairs(test_same | test_cross)) == 0
        # 投票者重叠（允许，但要报告）
        fold["voter_overlap_frac"] = float(
            len(set(df.loc[train, "voter_uniqueid"]) &
                set(df.loc[test_same | test_cross, "voter_uniqueid"])) /
            max(1, len(set(df.loc[test_same | test_cross, "voter_uniqueid"]))))

        summary.append(fold)
        print(f"  {test_city[:18]:<20} train={fold['n_train']:>7,} val={fold['n_val']:>6,} "
              f"test_same={fold['n_test_same']:>5,} test_cross={fold['n_test_cross']:>7,} "
              f"ok={fold["assert_T1_img_disjoint"] and fold["assert_val_img_disjoint"] and fold["assert_pair_disjoint"] and fold["assert_test_city_absent"]}")

    bad = [f for f in summary if not (f["assert_T1_img_disjoint"] and
                                      f["assert_val_img_disjoint"] and
                                      f["assert_pair_disjoint"] and
                                      f["assert_test_city_absent"])]
    print(f"\n断言失败折数: {len(bad)}")
    if bad:
        for f in bad[:5]:
            print("  ", {k: v for k, v in f.items() if k.startswith("assert")})

    (OUT / "logo_folds.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # 汇总
    tot_same = sum(f["n_test_same"] for f in summary)
    tot_cross = sum(f["n_test_cross"] for f in summary)
    print(f"\n合计测试票：同城 {tot_same:,}  跨城 {tot_cross:,}")
    print(f"中位 val 票数: {np.median([f['n_val'] for f in summary]):.0f}")
    print(f"投票者重叠比例中位: {np.median([f['voter_overlap_frac'] for f in summary]):.3f}")
    print(f"\n已写入 {OUT/'logo_folds.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
