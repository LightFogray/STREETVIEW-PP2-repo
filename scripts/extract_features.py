"""多骨干特征提取：ResNet-18 / ConvNeXt-Tiny / CLIP ViT-L/14。

用途
----
为「众包历史的边际价值是否随视觉模型质量下降」提供编码器阶梯。
三档设计（前两档 ImageNet-1k，第三档 CLIP 视觉塔）：

    resnet18        512-d   弱
    convnext_tiny   768-d   中（原稿使用）
    clip_vit_l14    768-d   强（与中档同宽，排除维度混淆）

用法
----
    python extract_features.py --selfcheck --backbone clip_vit_l14
    python extract_features.py --backbone resnet18
    python extract_features.py --backbone clip_vit_l14 --batch 32
"""
from __future__ import annotations

import argparse
import io
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

RAW = Path(r"D:\mycode\crime-temporal-spatio\data\raw\placepulse2")
Z = RAW / "place-pulse-2.0.zip"
DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
SIZE = 224

DIMS = {"resnet18": 512, "convnext_tiny": 768, "clip_vit_l14": 768}


def build_model(name: str, device):
    """返回 (model, forward_fn)。forward_fn: (B,3,224,224) -> (B,D)。"""
    if name == "resnet18":
        m = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()                      # 512-d after global pool
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        return m.to(device), lambda x: m(x)

    if name == "convnext_tiny":
        m = torchvision.models.convnext_tiny(
            weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        m = m.to(device)

        def fwd(x):
            f = m.features(x)
            f = m.classifier[0](f)                # LayerNorm2d
            return f.mean(dim=(-2, -1))

        return m, fwd

    if name == "clip_vit_l14":
        from transformers import CLIPModel
        m = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        m = m.to(device)
        def fwd(x):
            # transformers 5.x 返回 BaseModelOutputWithPooling；pooler_output 即 768 维投影嵌入
            out = m.get_image_features(pixel_values=x)
            return out.pooler_output if hasattr(out, "pooler_output") else out

        return m, fwd

    raise SystemExit(f"未知骨干: {name}")


def norm_for(name: str):
    return (CLIP_MEAN, CLIP_STD) if name.startswith("clip") else (IMAGENET_MEAN, IMAGENET_STD)


def preprocess(raw: bytes, mean, std) -> torch.Tensor:
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im = im.resize((SIZE, SIZE), Image.BICUBIC)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - mean) / std
    return torch.from_numpy(a.transpose(2, 0, 1))


def index_zip(zf: zipfile.ZipFile) -> dict[str, str]:
    out = {}
    for n in zf.namelist():
        if not (n.startswith("images/") and n.lower().endswith(".jpg")):
            continue
        parts = Path(n).stem.split("_")
        if len(parts) >= 3:
            out[parts[2]] = n
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="convnext_tiny",
                    choices=sorted(DIMS))
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    name = args.backbone
    dim = DIMS[name]
    batch = args.batch or (32 if name.startswith("clip") else 64)
    out = args.out or (DATA / f"features_{name}.npz")
    mean, std = norm_for(name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"backbone={name}  dim={dim}  batch={batch}  device={device}")

    model, fwd = build_model(name, device)
    print(f"  {'CLIP 视觉塔' if name.startswith('clip') else '参数量'} "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f} M")

    with zipfile.ZipFile(Z) as zf:
        idx = index_zip(zf)

    import pandas as pd
    df = pd.read_parquet(DATA / "pairs.parquet", columns=["left", "right"])
    ids = sorted(set(df["left"]) | set(df["right"]))
    have = [i for i in ids if i in idx]
    print(f"  可处理 {len(have):,} 张")
    if args.limit:
        have = have[: args.limit]
        print(f"  --limit -> {len(have)}")

    if args.selfcheck:
        print("\n=== 自检 ===")
        with zipfile.ZipFile(Z) as zf:
            x = torch.stack([preprocess(zf.read(idx[i]), mean, std)
                             for i in have[:8]]).to(device)
        with torch.no_grad():
            f1 = fwd(x)
            f2 = fwd(x)
        print(f"  1) 维度 {tuple(f1.shape)}  {'OK' if f1.shape[1] == dim else 'FAIL'}")
        print(f"  2) 确定性 {torch.equal(f1, f2)}")
        print(f"  3) 有限 {bool(torch.isfinite(f1).all())}")
        d = torch.cdist(f1, f1)
        off = d[~torch.eye(len(f1), dtype=bool, device=d.device)]
        print(f"  4) 成对距离 中位 {float(off.median()):.4f} 最小 {float(off.min()):.4f}")
        ok = (f1.shape[1] == dim and torch.equal(f1, f2)
              and bool(torch.isfinite(f1).all()) and float(off.min()) > 1e-3)
        print(f"\n  SELFCHECK {'OK' if ok else 'FAILED'}")
        return 0 if ok else 1

    print(f"\n开始提取 {len(have):,} 张 ...")
    feats = np.zeros((len(have), dim), dtype=np.float16)
    done = 0
    t0 = time.time()
    buf = []

    def flush():
        nonlocal done
        if not buf:
            return
        x = torch.stack(buf).to(device)
        with torch.no_grad():
            f = fwd(x)
        feats[done:done + len(buf)] = f.float().cpu().numpy().astype(np.float16)
        done += len(buf)
        buf.clear()

    with zipfile.ZipFile(Z) as zf:
        for k, i in enumerate(have):
            try:
                buf.append(preprocess(zf.read(idx[i]), mean, std))
            except Exception as exc:  # noqa: BLE001
                print(f"  读取失败 {i}: {exc}")
                buf.append(torch.zeros(3, SIZE, SIZE))
            if len(buf) >= batch:
                flush()
            if (k + 1) % 5000 == 0:
                el = time.time() - t0
                rate = (k + 1) / el
                print(f"  {k+1:,}/{len(have):,}  {rate:.0f} 张/秒  "
                      f"剩余 {(len(have)-k-1)/max(rate,1e-6)/60:.1f} 分钟")
    flush()

    print(f"\n完成 {done:,} 张，用时 {(time.time()-t0)/60:.1f} 分钟")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ids=np.array(have), features=feats)
    print(f"已写入 {out}  ({out.stat().st_size/1e6:.0f} MB)")
    print(f"统计 均值 {feats.astype(np.float32).mean():.4f}  "
          f"标准差 {feats.astype(np.float32).std():.4f}  "
          f"全零行 {int((np.abs(feats).sum(1)==0).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
