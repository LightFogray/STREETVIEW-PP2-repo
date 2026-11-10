"""Day 1：用 zip 自带的原始表构建权威图对表。

数据源（全部来自 place-pulse-2.0.zip，不用第三方加工版）：
    votes.tsv      _id, choice, left, right, study_id, timestamp, voter_uniqueid
    studies.tsv    _id -> study_question
    locations.tsv  _id -> place_id, loc.0(纬度), loc.1(经度), heading, pitch
    places.tsv     _id(place_id) -> place_name

产物：
    pairs.parquet   每票一行，含两图 ID、城市、坐标、choice、voter、时间
    build_summary.json

用法：
    python build_tables.py
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(r"<PROJECT>/data/raw/placepulse2")
T = RAW / "from_zip"
Z = RAW / "place-pulse-2.0.zip"
OUT = Path(r"<PROJECT>/STREETVIEW-PP2/data")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    print("1. 读取原始表")
    st = pd.read_csv(T / "studies.tsv", sep="\t")
    loc = pd.read_csv(T / "locations.tsv", sep="\t")
    pl = pd.read_csv(T / "places.tsv", sep="\t")
    v = pd.read_csv(T / "votes.tsv", sep="\t")
    print(f"   votes {len(v):,}  studies {len(st)}  locations {len(loc):,}  places {len(pl)}")

    print("\n2. 关联问题文本并筛出 safer")
    qmap = dict(zip(st["_id"], st["study_question"]))
    v["question"] = v["study_id"].map(qmap)
    s = v[v["question"] == "safer"].copy()
    print(f"   safer 全部票: {len(s):,}")
    s = s[s["choice"].isin(["left", "right", "equal"])]
    s = s[s["left"].notna() & s["right"].notna() & (s["left"] != s["right"])]
    print(f"   有效票（choice 合法、两图不同）: {len(s):,}")

    print("\n3. 关联图像元数据")
    meta = loc.set_index("_id")[["place_id", "loc.0", "loc.1", "heading", "pitch"]]
    meta.columns = ["place_id", "lat", "lon", "heading", "pitch"]
    pmap = dict(zip(pl["_id"], pl["place_name"]))
    meta["city"] = meta["place_id"].map(pmap)
    print(f"   有城市的图像: {meta['city'].notna().sum():,} / {len(meta):,}")

    for side in ("left", "right"):
        joined = meta.reindex(s[side].values)
        s[f"{side}_city"] = joined["city"].values
        s[f"{side}_lat"] = joined["lat"].values
        s[f"{side}_lon"] = joined["lon"].values
    miss_l = int(s["left_city"].isna().sum())
    miss_r = int(s["right_city"].isna().sum())
    print(f"   左图缺城市 {miss_l:,}   右图缺城市 {miss_r:,}")

    print("\n4. 只保留两图都有城市与坐标的票")
    keep = (s["left_city"].notna() & s["right_city"].notna()
            & s["left_lat"].notna() & s["right_lat"].notna())
    s = s[keep].copy()
    print(f"   保留 {len(s):,} 票")

    print("\n5. 图像级比较次数（用于可靠性变量）")
    both = pd.concat([s["left"], s["right"]], ignore_index=True)
    n_cmp = both.value_counts()
    s["n_left"] = s["left"].map(n_cmp).astype(int)
    s["n_right"] = s["right"].map(n_cmp).astype(int)
    print(f"   涉及图像 {len(n_cmp):,}   每图中位比较数 {n_cmp.median():.0f}")

    print("\n6. 与 zip 内图像是否对得上")
    with zipfile.ZipFile(Z) as z:
        names = {Path(n).stem.split("_")[2]
                 for n in z.namelist()
                 if n.startswith("images/") and n.lower().endswith(".jpg")}
    ref = set(n_cmp.index)
    hit = len(ref & names)
    print(f"   票里引用的图像 {len(ref):,}  在 zip 中存在 {hit:,} "
          f"= {hit/len(ref)*100:.2f}%")

    print("\n7. 同城 / 跨城结构")
    s["same_city"] = s["left_city"] == s["right_city"]
    n_same = int(s["same_city"].sum())
    print(f"   同城 {n_same:,} ({n_same/len(s)*100:.2f}%)   "
          f"跨城 {len(s)-n_same:,}")

    print("\n8. 三类标签")
    cc = s["choice"].value_counts(normalize=True)
    print("   " + "  ".join(f"{k}={v:.4f}" for k, v in cc.items()))
    for name, m in (("同城", s["same_city"]), ("跨城", ~s["same_city"])):
        c = s[m]["choice"].value_counts(normalize=True)
        print(f"   {name}: equal={c.get('equal',0):.4f} "
              f"left={c.get('left',0):.4f} right={c.get('right',0):.4f}")

    print("\n9. 每城同城票数（决定 LOGO 候选城）")
    same = s[s["same_city"]]
    per_city = same["left_city"].value_counts()
    print(f"   城市数 {len(per_city)}   中位 {per_city.median():.0f}")
    print(f"   >=300 票的城市 {int((per_city>=300).sum())}   "
          f">=500 {int((per_city>=500).sum())}")
    for c, k in per_city.head(12).items():
        n_either = int(((s["left_city"] == c) | (s["right_city"] == c)).sum())
        print(f"     {c[:22]:<24}{k:>7,} 同城    {n_either:>8,} 含本城")

    print("\n10. 投票者结构")
    vc = s["voter_uniqueid"].value_counts()
    vcities = s.groupby("voter_uniqueid")["left_city"].nunique()
    print(f"   投票者 {len(vc):,}   每人票数中位 {vc.median():.0f}  max {vc.max():,}")
    print(f"   跨 >1 城的投票者 {float((vcities>1).mean())*100:.1f}%")

    cols = ["left", "right", "choice", "study_id", "voter_uniqueid", "timestamp",
            "left_city", "right_city", "left_lat", "left_lon", "right_lat", "right_lon",
            "n_left", "n_right", "same_city"]
    out = s[cols].reset_index(drop=True)
    out.to_parquet(OUT / "pairs.parquet", index=False)
    print(f"\n已写入 {OUT/'pairs.parquet'}  ({len(out):,} 行)")

    summary = {
        "n_votes_raw": int(len(v)),
        "n_safer_valid": int(len(out)),
        "n_images": int(len(n_cmp)),
        "image_zip_hit_rate": hit / len(ref),
        "n_same_city": n_same,
        "same_city_frac": n_same / len(out),
        "choice_overall": {k: float(x) for k, x in cc.items()},
        "cities_with_same_city_votes": int(len(per_city)),
        "cities_ge_300": int((per_city >= 300).sum()),
        "cities_ge_500": int((per_city >= 500).sum()),
        "n_voters": int(len(vc)),
        "voter_multi_city_frac": float((vcities > 1).mean()),
    }
    (OUT / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
