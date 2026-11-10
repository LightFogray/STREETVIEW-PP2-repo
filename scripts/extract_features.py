"""Day 2：从 zip 直接读取图像，用冻结的 ConvNeXt-Tiny 提取特征。

设计取舍
--------
- **不解压**。111k 个小文件在 NTFS 上读写很慢，而 zipfile 支持按名随机读。
- **特征只算一次**，后续 B0–B4 全部复用。
- 输出 float16，111k×768 ≈ 170 MB。

自检项（--selfcheck）
--------------------
1. 模型输出维度正确
2. 同一张图两次提取完全一致（确定性）
3. 不同图产生不同特征
4. 特征有限、非常数
5. ID 覆盖率

用法：
    python extract_features.py --selfcheck
    python extract_features.py --out data/features_convnext_tiny.npz
"""
from __future__ import annotations

import argparse
import io
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

RAW = Path(r"<PROJECT>/data/raw/placepulse2")
Z = RAW / "place-pulse-2.0.zip"
DATA = Path(r"<PROJECT>/STREETVIEW-PP2/data")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 224


def build_model(device):
    m = torchvision.models.convnext_tiny(
        weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m.to(device)


def forward_features(model, x):
    """(B,3,224,224) -> (B,768) 池化后特征。"""
    f = model.features(x)
    f = model.classifier[0](f)          # LayerNorm2d
    return f.mean(dim=(-2, -1))


def preprocess(raw: bytes) -> torch.Tensor:
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im = im.resize((SIZE, SIZE), Image.BICUBIC)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(a.transpose(2, 0, 1))


def index_zip(zf: zipfile.ZipFile) -> dict[str, str]:
    """image_id -> zip 成员名。文件名格式 lat_lon_id_City.JPG"""
    out = {}
    for n in zf.namelist():
        if not (n.startswith("images/") and n.lower().endswith(".jpg")):
            continue
        stem = Path(n).stem
        parts = stem.split("_")
        if len(parts) >= 3:
            out[parts[2]] = n
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", type=Path, default=DATA / "features_convnext_tiny.npz")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    print("加载模型（首次会下载权重，约 110 MB）...")
    model = build_model(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  参数量 {n_par/1e6:.2f} M")

    with zipfile.ZipFile(Z) as zf:
        idx = index_zip(zf)
    print(f"zip 内图像 {len(idx):,}")

    # 需要提取的 ID：pairs 表里出现过的
    import pandas as pd
    df = pd.read_parquet(DATA / "pairs.parquet", columns=["left", "right"])
    ids = sorted(set(df["left"]) | set(df["right"]))
    have = [i for i in ids if i in idx]
    print(f"pairs 中引用 {len(ids):,}，zip 中可找到 {len(have):,} "
          f"({len(have)/len(ids)*100:.2f}%)")
    if args.limit:
        have = have[: args.limit]
        print(f"  --limit 生效，只处理 {len(have)}")

    # ---------- 自检 ----------
    if args.selfcheck:
        print("\n=== 自检 ===")
        with zipfile.ZipFile(Z) as zf:
            sample = have[:8]
            x = torch.stack([preprocess(zf.read(idx[i])) for i in sample]).to(device)
        with torch.no_grad():
            f1 = forward_features(model, x)
            f2 = forward_features(model, x)
        print(f"  1) 输出维度 {tuple(f1.shape)}  {'OK' if f1.shape[1]==768 else 'FAIL'}")
        print(f"  2) 两次提取完全一致: {torch.equal(f1, f2)}")
        print(f"  3) 有限: {bool(torch.isfinite(f1).all())}")
        # 不同图之间应不同
        d = torch.cdist(f1, f1)
        off = d[~torch.eye(len(f1), dtype=bool, device=d.device)]
        print(f"  4) 成对距离 中位 {float(off.median()):.4f}  最小 {float(off.min()):.4f}")
        print(f"     特征标准差（跨样本） {float(f1.std(0).mean()):.4f}")
        ok = (f1.shape[1] == 768 and torch.equal(f1, f2)
              and bool(torch.isfinite(f1).all()) and float(off.min()) > 1e-3)
        print(f"\n  SELFCHECK {'OK' if ok else 'FAILED'}")
        if not ok:
            return 1
        print("  自检通过，退出（去掉 --selfcheck 才跑全量）")
        return 0

    # ---------- 全量提取 ----------
    print(f"\n开始提取 {len(have):,} 张 ...")
    feats = np.zeros((len(have), 768), dtype=np.float16)
    done = 0
    t0 = time.time()
    buf, buf_ids = [], []

    def flush():
        nonlocal done
        if not buf:
            return
        x = torch.stack(buf).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                             enabled=device.type == "cuda"):
            f = forward_features(model, x)
        feats[done:done + len(buf)] = f.float().cpu().numpy().astype(np.float16)
        done += len(buf)
        buf.clear()

    with zipfile.ZipFile(Z) as zf:
        for k, i in enumerate(have):
            try:
                buf.append(preprocess(zf.read(idx[i])))
            except Exception as exc:  # noqa: BLE001
                print(f"  读取失败 {i}: {exc}")
                buf.append(torch.zeros(3, SIZE, SIZE))
            if len(buf) >= args.batch:
                flush()
            if (k + 1) % 5000 == 0:
                el = time.time() - t0
                rate = (k + 1) / el
                print(f"  {k+1:,}/{len(have):,}  {rate:.0f} 张/秒  "
                      f"剩余 {(len(have)-k-1)/max(rate,1e-6)/60:.1f} 分钟")
    flush()

    print(f"\n完成 {done:,} 张，用时 {(time.time()-t0)/60:.1f} 分钟")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ids=np.array(have), features=feats)
    print(f"已写入 {args.out}  ({args.out.stat().st_size/1e6:.0f} MB)")
    print(f"特征统计: 均值 {feats.astype(np.float32).mean():.4f}  "
          f"标准差 {feats.astype(np.float32).std():.4f}  "
          f"全零行 {int((np.abs(feats).sum(1)==0).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
