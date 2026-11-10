"""图像与坐标可追溯性核验 v2。

v1 的两处判据错误已修正，并确认了两个真实数据缺陷：

缺陷 1（严重）：经纬度列互换
    Amsterdam: lat_left=4.910, long_left=52.336  → 真实 (52.37, 4.90)
    Chicago  : lat_left=-87.705, long_left=41.845 → 真实 (41.88, -87.63)
    即 **`long_*` 列装的是纬度，`lat_*` 列装的是经度**。
    按列名直接读取会把 Amsterdam 放到德国北部海面、把 Chicago 放到大西洋。

缺陷 2（影响核验方式）：图像 ID 不是 Google panoid
    `left`/`right`/`place_id_*` 全部是 24 位纯 hex（MongoDB ObjectId 格式），
    发布记录使用数据集内部标识。这份投票表没有到 Google panoid 的映射。
    可依托有来源的图像档案、位置索引或文件名清单匹配；名称并不能证明像素身份。

因此本脚本现在做：
    A. 自动检测并修正坐标互换，重新做一致性检查
    B. 生成"位置核验"清单（修正后的坐标 + Google Maps 链接），用于人工抽查城市与地点
    C. 生成"图像核验"清单（ID 列表），等 images.zip 到位后按文件名比对

用法：
    python verify_pp2_images.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(r"<PROJECT>/data/raw/placepulse2/figshare_11859993_v1")
OUT = Path(r"<PROJECT>/STREETVIEW-PP2/outputs")
MAPS_URL = "https://www.google.com/maps/@{lat},{lon},17z/data=!3m1!1e3"

# 用于验证坐标方向的已知城市中心
TRUTH = {"Amsterdam": (52.37, 4.90), "Atlanta": (33.75, -84.39),
         "Berlin": (52.52, 13.40), "Tokyo": (35.68, 139.73),
         "Rio De Janeiro": (-22.91, -43.17), "Chicago": (41.88, -87.63)}


def detect_swap(df: pd.DataFrame) -> tuple[bool, dict]:
    """判断 lat_/long_ 是否互换：用已知城市中心做对照。"""
    votes = {}
    for c, (tlat, tlon) in TRUTH.items():
        m = df[df["place_name_left"].astype(str).str.strip() == c].dropna(
            subset=["lat_left", "long_left"])
        if len(m) < 10:
            continue
        a, b = m["lat_left"].median(), m["long_left"].median()
        d_as_is = abs(a - tlat) + abs(b - tlon)          # 按列名解释
        d_swap = abs(b - tlat) + abs(a - tlon)           # 互换解释
        votes[c] = {"as_is_err": round(float(d_as_is), 3),
                    "swapped_err": round(float(d_swap), 3)}
    if not votes:
        return False, {}
    swap_wins = sum(v["swapped_err"] < v["as_is_err"] for v in votes.values())
    return swap_wins >= max(1, len(votes) // 2), votes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["Atlanta", "Berlin", "Tokyo"])
    ap.add_argument("--per-city", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("读取投票表 ...")
    cols = ["left", "right", "choice", "study_question", "voter_uniqueid",
            "place_name_left", "place_name_right", "lat_left", "long_left",
            "lat_right", "long_right"]
    df = pd.read_csv(D / "votes_clean.csv", usecols=cols, low_memory=False)
    v = df[df["study_question"].astype(str).str.contains("safe", case=False, na=False)]
    v = v[v["choice"].isin(["left", "right", "equal"])]
    v = v[v["left"].notna() & v["right"].notna() & (v["left"] != v["right"])]
    print(f"  safer 有效行 {len(v):,}")

    print("\n" + "=" * 72)
    print("缺陷 1：经纬度列互换检测")
    print("=" * 72)
    swapped, votes = detect_swap(v)
    for c, e in votes.items():
        print(f"  {c:<16} 按列名误差={e['as_is_err']:>8.3f}   互换后误差={e['swapped_err']:>8.3f}")
    print(f"  => 判定为互换: {swapped}")

    # ---------- 构造图像表（应用修正）----------
    lat_l = v["lat_left"] if not swapped else v["long_left"]
    lon_l = v["long_left"] if not swapped else v["lat_left"]
    lat_r = v["lat_right"] if not swapped else v["long_right"]
    lon_r = v["long_right"] if not swapped else v["lat_right"]
    left = pd.DataFrame({"panoid": v["left"], "city": v["place_name_left"],
                         "lat": lat_l, "lon": lon_l})
    right = pd.DataFrame({"panoid": v["right"], "city": v["place_name_right"],
                          "lat": lat_r, "lon": lon_r})
    img = pd.concat([left, right], ignore_index=True)
    img["lat"] = pd.to_numeric(img["lat"], errors="coerce")
    img["lon"] = pd.to_numeric(img["lon"], errors="coerce")
    n_cmp = img["panoid"].value_counts()
    uniq = img.drop_duplicates("panoid")

    print("\n" + "=" * 72)
    print("A. 修正后的一致性检查")
    print("=" * 72)
    fails = []
    bad = uniq[(uniq["lat"].abs() > 90) | (uniq["lon"].abs() > 180)]
    print(f"  1) 坐标越界（修正后）: {len(bad):,}")
    if len(bad):
        fails.append(f"{len(bad)} 越界")
    multi = int((img.groupby("panoid")["city"].nunique() > 1).sum())
    print(f"  2) 同一 ID 跨城: {multi:,}")
    if multi:
        fails.append(f"{multi} 个 ID 跨城")
    print(f"  3) 坐标缺失: {uniq['lat'].isna().mean()*100:.2f}%")

    print("\n" + "=" * 72)
    print("缺陷 2：图像 ID 格式")
    print("=" * 72)
    ids = uniq["panoid"].astype(str)
    is_hex24 = ids.str.fullmatch(r"[0-9a-f]{24}").mean()
    print(f"  24 位纯 hex 占比: {is_hex24*100:.2f}%")
    print(f"  以 'Cn' 开头（Google panoid 特征）占比: {ids.str.startswith('Cn').mean()*100:.2f}%")
    print("  => 这些是数据集内部的 ObjectId，**不能**直接用于 Google 街景 URL。")
    print("     需要来源可核实的档案或映射；仅匹配文件名不验证历史街景像素。")

    # ---------- B. 位置核验清单 ----------
    print("\n" + "=" * 72)
    print("B. 位置核验清单（城市与坐标是否相符，不涉及图像内容）")
    print("=" * 72)
    rows = []
    for city in args.cities:
        sub = uniq[uniq["city"] == city].dropna(subset=["lat", "lon"]).copy()
        if len(sub) == 0:
            print(f"  {city}: 无数据")
            continue
        sub["n_cmp"] = sub["panoid"].map(n_cmp)
        sub = sub.sort_values("n_cmp").reset_index(drop=True)
        k = min(args.per_city, len(sub))
        idx = np.unique(np.linspace(0, len(sub) - 1, k).round().astype(int))
        pick = sub.iloc[idx]
        print(f"  {city}: 唯一图像 {len(sub):,}，抽 {len(pick)} 个")
        for _, r in pick.iterrows():
            rows.append({
                "city": city, "image_id": r["panoid"],
                "lat_fixed": round(float(r["lat"]), 6),
                "lon_fixed": round(float(r["lon"]), 6),
                "n_comparisons": int(r["n_cmp"]),
                "maps_url": MAPS_URL.format(lat=r["lat"], lon=r["lon"]),
                "人工: 地图位置在该城市?": "",
                "人工: 与城市名相符?": "",
                "备注": "",
            })
    loc_df = pd.DataFrame(rows)
    loc_df.drop(columns=["maps_url"]).to_csv(
        OUT / "location_checklist.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "location_checklist.xlsx", engine="openpyxl") as w:
        loc_df.drop(columns=["maps_url"]).to_excel(w, index=False, sheet_name="位置核验")
        ws = w.sheets["位置核验"]
        ws.insert_cols(6)
        ws.cell(row=1, column=6, value="地图链接")
        for i, u in enumerate(loc_df["maps_url"], start=2):
            c = ws.cell(row=i, column=6, value="打开地图")
            c.hyperlink = u
        for col, wd in zip("ABCDEFGHI", [12, 26, 12, 13, 13, 12, 22, 20, 18]):
            ws.column_dimensions[col].width = wd
    print(f"  已生成 location_checklist.csv / .xlsx（{len(loc_df)} 行）")

    # ---------- C. 图像核验清单 ----------
    print("\n" + "=" * 72)
    print("C. 图像核验清单（等 images.zip 到位后用）")
    print("=" * 72)
    image_ids = uniq[["panoid", "city"]].copy()
    image_ids["n_comparisons"] = image_ids["panoid"].map(n_cmp)
    image_ids.to_csv(OUT / "image_ids_for_zip_check.csv", index=False, encoding="utf-8-sig")
    print(f"  已导出全部 {len(image_ids):,} 个 image_id 与城市")
    print("  用 check_image_id_coverage.py 直接读取 ZIP/TAR 成员名，无需解压。")
    print("  ≥90% 是总体 ID 覆盖门槛；另核逐城缺失、像素可读性与原始视图来源。")

    summary = {"latlon_swapped": bool(swapped), "swap_evidence": votes,
               "n_unique_image_ids": int(len(uniq)),
               "id_is_hex24_frac": float(is_hex24),
               "coord_out_of_range_after_fix": int(len(bad)),
               "id_multi_city": multi,
               "checks_failed": fails}
    (OUT / "image_verification_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n汇总已写入 image_verification_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
