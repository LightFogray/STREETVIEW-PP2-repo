"""投稿前的完整数据核查。

逐条核对稿件里出现的每个数字，并检查数据本身的结构性问题。
任何一条 FAIL 都必须先解决再投稿。

用法：
    python audit_all_numbers.py
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(r"<PROJECT>/data/raw/placepulse2")
T = RAW / "from_zip"
Z = RAW / "place-pulse-2.0.zip"
D = Path(r"<PROJECT>/STREETVIEW-PP2/data")
O = Path(r"<PROJECT>/STREETVIEW-PP2/outputs")

results = []


def check(name: str, got, expected, tol=0) -> bool:
    if isinstance(expected, (int, np.integer)) and tol == 0:
        ok = int(got) == int(expected)
    elif isinstance(expected, float):
        ok = abs(float(got) - expected) <= tol
    else:
        ok = got == expected
    results.append((name, got, expected, ok))
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name:<46} 实测={got}  期望={expected}")
    return ok


def main() -> int:
    print("=" * 84)
    print("A. 原始档案计数")
    print("=" * 84)
    st = pd.read_csv(T / "studies.tsv", sep="\t")
    loc = pd.read_csv(T / "locations.tsv", sep="\t")
    pl = pd.read_csv(T / "places.tsv", sep="\t")
    v = pd.read_csv(T / "votes.tsv", sep="\t")
    check("原始投票行数", len(v), 1_555_561)
    check("studies 行数", len(st), 6)
    check("locations 行数", len(loc), 111_390)
    check("places（城市）数", len(pl), 56)

    qmap = dict(zip(st["_id"], st["study_question"]))
    v["q"] = v["study_id"].map(qmap)
    s = v[v["q"] == "safer"]
    check("safer 全部票", len(s), 506_165)
    s2 = s[s["choice"].isin(["left", "right", "equal"])]
    s2 = s2[s2["left"].notna() & s2["right"].notna() & (s2["left"] != s2["right"])]
    check("safer 有效票", len(s2), 506_165)

    tie_rate = (s2["choice"] == "equal").mean() * 100
    check("平局率(%)", round(float(tie_rate), 2), 12.47, tol=0.02)

    # 重复票检测
    dup_key = s2.assign(_k=s2["left"] + "|" + s2["right"] + "|" +
                        s2["voter_uniqueid"].astype(str) + "|" + s2["timestamp"].astype(str))
    n_dup = int(len(dup_key) - dup_key["_k"].nunique())
    print(f"  [INFO] 完全重复的投票记录（同图对+同投票者+同时间戳）: {n_dup:,}")
    results.append(("重复投票记录", n_dup, 0, n_dup == 0))

    print("\n" + "=" * 84)
    print("B. 图像与特征覆盖")
    print("=" * 84)
    with zipfile.ZipFile(Z) as z:
        names = [n for n in z.namelist() if n.startswith("images/") and n.lower().endswith(".jpg")]
    zip_ids = {Path(n).stem.split("_")[2] for n in names}
    check("zip 内图像数", len(names), 110_988)
    check("zip 内唯一 ID", len(zip_ids), 110_988)

    df = pd.read_parquet(D / "pairs.parquet")
    vote_ids = set(df["left"]) | set(df["right"])
    check("票中引用的唯一图像", len(vote_ids), 111_389)
    hit = len(vote_ids & zip_ids)
    check("图像可取得率(%)", round(hit / len(vote_ids) * 100, 2), 99.64, tol=0.02)

    zf = np.load(D / "features_convnext_tiny.npz")
    feat_ids = set(zf["ids"].tolist())
    check("有特征的图像数", len(feat_ids), 110_987)
    check("特征维度", zf["features"].shape[1], 768)

    dfm = df[df["left"].isin(feat_ids) & df["right"].isin(feat_ids)].copy()
    check("两端都有特征的票", len(dfm), 503_291)
    _idx = {k: i for i, k in enumerate(zf["ids"].tolist())}
    dfm["_ia"] = dfm["left"].map(_idx).astype(np.int32)
    dfm["_ib"] = dfm["right"].map(_idx).astype(np.int32)

    print("\n" + "=" * 84)
    print("C. 划分与泄漏")
    print("=" * 84)
    folds = json.loads((D / "splits" / "logo_folds.json").read_text(encoding="utf-8"))
    check("折数", len(folds), 15)

    bad = []
    t2_total = 0
    uniq_t2 = set()
    for f in folds:
        c, val = f["test_city"], f["val_city"]
        in_t = (dfm["left_city"] == c) | (dfm["right_city"] == c)
        in_v = (dfm["left_city"] == val) | (dfm["right_city"] == val)
        tr = dfm[(~in_t) & (~in_v)]
        te_same = dfm[in_t & dfm["same_city"]]
        te_cross = dfm[in_t & (~dfm["same_city"])]
        te_all = pd.concat([te_same, te_cross], ignore_index=True)
        # 测试城图像绝不在训练中
        tr_imgs = set(tr["left"]) | set(tr["right"])
        te_imgs = set(te_same["left"]) | set(te_same["right"])
        if tr_imgs & te_imgs:
            bad.append((c, "T1 图像泄漏"))
        # val 图像绝不在训练中
        # 注意：必须只取"城市标签为 val"的那一端，不能把它们的对手也算进来
        va_imgs = (set(dfm.loc[dfm["left_city"] == val, "left"])
                   | set(dfm.loc[dfm["right_city"] == val, "right"]))
        if tr_imgs & va_imgs:
            bad.append((c, "val 图像泄漏"))
        # 图对不跨集合
        def pk(d):
            return set(zip(np.minimum(d["left"], d["right"]),
                           np.maximum(d["left"], d["right"])))
        if pk(tr) & pk(te_all):
            bad.append((c, "图对跨集合"))
        t2_total += len(te_cross)
        for r in te_cross.itertuples():
            uniq_t2.add(hashlib.md5(f"{r.left}|{r.right}|{r.voter_uniqueid}|{r.timestamp}".encode()).hexdigest())

    check("泄漏问题数", len(bad), 0)
    if bad:
        for b in bad[:5]:
            print("     ", b)
    check("T2 折记录总数", t2_total, 437_587)
    check("T2 不同投票行数", len(uniq_t2), 343_573)

    print("\n" + "=" * 84)
    print("D. 历史可得性")
    print("=" * 84)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from anchor_history_fusion import history_quality
    idx = {k: i for i, k in enumerate(zf["ids"].tolist())}
    n_img = len(idx)
    warm, ge5, tot = 0, 0, 0
    for f in folds:
        c, val = f["test_city"], f["val_city"]
        in_t = (dfm["left_city"] == c) | (dfm["right_city"] == c)
        in_v = (dfm["left_city"] == val) | (dfm["right_city"] == val)
        tr = dfm[(~in_t) & (~in_v)]
        te = dfm[in_t & (~dfm["same_city"])]
        _, n = history_quality(tr, n_img)
        mx = np.maximum(n[[idx[x] for x in te["left"]]], n[[idx[x] for x in te["right"]]])
        warm += int((mx > 0).sum()); ge5 += int((mx >= 5).sum()); tot += len(te)
    check("有历史比例(%)", round(warm / tot * 100, 2), 96.28, tol=0.02)
    check("至少 5 次比较比例(%)", round(ge5 / tot * 100, 2), 86.26, tol=0.02)

    print("\n" + "=" * 84)
    print("E. 稿件表格数字")
    print("=" * 84)
    eq = pd.read_csv(O / "equal_interaction_screen" / "metrics.csv")
    e = eq[eq["arm"] == "diff"]
    check("T1 视觉 logloss(4ep)", round(float(e[e["test"] == "same"]["logloss"].mean()), 5), 0.97479, tol=5e-4)
    check("T2 视觉 logloss(4ep)", round(float(e[e["test"] == "cross"]["logloss"].mean()), 5), 0.90560, tol=5e-4)
    check("T1 平局 AP", round(float(e[e["test"] == "same"]["equal_ap"].mean()), 5), 0.20651, tol=5e-4)
    check("T2 平局 AP", round(float(e[e["test"] == "cross"]["equal_ap"].mean()), 5), 0.14801, tol=5e-4)

    au = pd.read_csv(O / "anchor_control_audit" / "per_fold.csv")
    check("Table2 A 视觉基线", round(float(au["A_b2"].mean()), 6), 0.905491, tol=5e-6)
    check("Table2 C 加真实历史", round(float(au["C_b2_plus_hist"].mean()), 6), 0.904930, tol=5e-6)

    print("\n" + "=" * 84)
    print("F. 三种子稳定性")
    print("=" * 84)
    seeds = [pd.read_csv(O / "bt_history" / f"per_fold_seed{s}.csv") for s in (0, 1, 2)]
    A = [float(x["A_b2"].mean()) for x in seeds]
    Cn = [float(x["C_net"].mean()) for x in seeds]
    Cb = [float(x["C_bt"].mean()) for x in seeds]
    check("三种子 A 均值", round(float(np.mean(A)), 6), 0.905271, tol=2e-6)
    check("三种子 A 的 SD", round(float(np.std(A, ddof=1)), 6), 0.000224, tol=2e-6)
    dn = [c - a for c, a in zip(Cn, A)]
    db = [c - a for c, a in zip(Cb, A)]
    check("净胜分增益均值", round(float(np.mean(dn)), 6), -0.000561, tol=2e-6)
    check("净胜分增益种子 SD", round(float(np.std(dn, ddof=1)), 6), 0.000020, tol=2e-6)
    check("BT 增益均值", round(float(np.mean(db)), 6), -0.000439, tol=2e-6)

    print("\n" + "=" * 84)
    n_pass = sum(1 for r in results if r[3])
    print(f"总计 {len(results)} 项检查，通过 {n_pass}，失败 {len(results)-n_pass}")
    fail = [r for r in results if not r[3]]
    if fail:
        print("\n失败项：")
        for name, got, exp, _ in fail:
            print(f"  - {name}: 实测 {got}  期望 {exp}")
    print("=" * 84)
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
