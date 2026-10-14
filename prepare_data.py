"""
数据预处理：两种数据源 -> 统一 npy 格式
==========================================
模式 A（论文原数据集，Fig.3.zip 解压后预期结构）：
  目录下每个样本一对多页 tif：
    <name>_img.tif   -> (D,H,W) 原图 volume（如 800x800x800）
    <name>_seg.tif   -> (D,H,W) 人工标注 GT（0/255 二值）
  输出 <name>_img.npy / <name>_seg.npy + splits.json（按样本 4-fold 划分）

模式 B（本机 OCT_SweatGland BMP 序列）：
  <src>/images/<prefix>-DA-<frame>.bmp + <src>/labels/<prefix>-DA-<frame>-sg.bmp
  按 prefix + 帧号连续性分组，每段连续帧组成一个 (D,H,W) volume。
  1800x500 B-scan：横向保留，深度方向按 meta 里的组织区域裁 288 高（或自定义 --depth-roi）。

用法：
  python prepare_data.py --mode tif   --src <Fig.3解压目录> --out data/volumes
  python prepare_data.py --mode bmp   --src D:/zust/OCT_SweatGland --out data/volumes
  python prepare_data.py --mode synth --out data/volumes --n 8      # 合成数据冒烟用
"""

import os
import re
import json
import argparse
import numpy as np


def group_bmp_sequences(src_images, src_labels):
    """BMP 序列按 (prefix, 连续帧段) 分组。返回 [(prefix, seg_id, [frame,...]), ...]
    保留 image 与 label 同时存在的全部帧（尺寸异常的帧在读取时 resize 到 1800x500，
    实测这些帧长宽比均为 3.6，是等比例缩小的正常帧）。"""
    pattern = re.compile(r"(.+)-DA-(\d+)(-sg)?\.(bmp|png)$")
    frames = {}
    for d in (src_images, src_labels):
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            m = pattern.match(f)
            if m:
                prefix, frame = m.group(1), int(m.group(2))
                frames.setdefault(prefix, set()).add(frame)

    # 尺寸异常帧仅警告（读取时 resize），不跳过——跳过会打断连续段
    from PIL import Image
    odd = []
    for d in (src_images, src_labels):
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            m = pattern.match(f)
            if m:
                try:
                    if Image.open(os.path.join(d, f)).size != (1800, 500):
                        odd.append(f)
                except Exception:
                    pass
    if odd:
        print(f"  [warn] {len(odd)} 帧非标准尺寸，读取时将 resize 到 1800x500: {sorted(odd)}")

    groups = []
    for prefix in sorted(frames):
        fr = sorted(frames[prefix])
        if not fr:
            continue
        seg, start, prev = [fr[0]], fr[0], fr[0]
        for x in fr[1:]:
            if x == prev + 1:
                seg.append(x)
            else:
                groups.append((prefix, start, seg))
                seg, start = [x], x
            prev = x
        groups.append((prefix, start, seg))
    return groups


def save_npy_pair(vol, seg, out_dir, name):
    img_p = os.path.join(out_dir, f"{name}_img.npy")
    seg_p = os.path.join(out_dir, f"{name}_seg.npy")
    np.save(img_p, vol.astype(np.uint8))
    np.save(seg_p, (seg > 0).astype(np.uint8))
    return img_p


def make_splits(items, n_folds=4, out_dir="."):
    """按样本 n-fold 划分（论文 Sec III-B: train:val = 3:1）。
    样本数不足时自动降为 len(items) 折（留一法），避免空折。"""
    n_folds = min(n_folds, len(items))
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(items))
    folds = [idx[i::n_folds].tolist() for i in range(n_folds)]
    split = {"items": [os.path.basename(x) for x in items], "folds": folds}
    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump(split, f, indent=1)
    return split


def mode_tif(src, out, n_folds, manifest=None):
    """src 目录下按 <name>_img.tif / <name>_seg.tif 配对；
    或传 manifest: JSON 文件路径，内容 {name: {"img": 路径, "seg": 路径}}（无需复制原文件）"""
    import tifffile
    os.makedirs(out, exist_ok=True)
    pairs = []
    if manifest:
        with open(manifest, encoding="utf-8") as f:
            pairs = [(name, d["img"], d["seg"]) for name, d in json.load(f).items()]
    else:
        names = sorted(set(
            os.path.splitext(f)[0].replace("_img", "").replace("_seg", "")
            for f in os.listdir(src) if f.lower().endswith((".tif", ".tiff"))))
        pairs = [(n, os.path.join(src, n + "_img.tif"), os.path.join(src, n + "_seg.tif")) for n in names]
    items = []
    for name, ip, sp in pairs:
        if not (os.path.exists(ip) and os.path.exists(sp)):
            print(f"  [skip] {name}: 缺少 img/seg 文件")
            continue
        vol, seg = tifffile.imread(ip), tifffile.imread(sp)
        if vol.ndim == 4:      # 若 tif 是 (C,D,H,W) 取首通道
            vol = vol[0]
        if seg.ndim == 4:
            seg = seg[0]
        items.append(save_npy_pair(vol, (seg > 0).astype(np.uint8) * 255, out, name))
        print(f"  {name}: volume {vol.shape}  fg={float((seg>0).mean()):.4f}")
    if not items:
        raise SystemExit("没有可用配对，检查 manifest/src")
    make_splits(items, n_folds, out)
    print(f"total volumes: {len(items)}, splits.json written")


def mode_bmp(src, out, n_folds, depth_roi=288, depth_center=140):
    """depth_roi: 深度方向裁剪高度; depth_center: 裁剪窗中心（默认按汗腺分布带居中）
    注意布局：PIL 读出 shape = (H=深度500, W=横向1800)，堆叠后 vol = (D, 500, 1800)"""
    from PIL import Image
    os.makedirs(out, exist_ok=True)
    groups = group_bmp_sequences(os.path.join(src, "images"), os.path.join(src, "labels"))
    items = []
    for prefix, start, seg_frames in groups:
        imgs, segs = [], []
        for fr in seg_frames:
            ip = os.path.join(src, "images", f"{prefix}-DA-{fr}.bmp")
            sp = os.path.join(src, "labels", f"{prefix}-DA-{fr}-sg.bmp")
            imgs.append(np.array(Image.open(ip).convert("L").resize((1800, 500))))
            segs.append((np.array(Image.open(sp).convert("L").resize((1800, 500))) > 0).astype(np.uint8) * 255)
        vol = np.stack(imgs)     # (D, H=深度500, W=横向1800)
        seg = np.stack(segs)
        # 深度方向（H）裁 ROI
        h_total = vol.shape[1]
        if h_total > depth_roi:
            c = int(np.clip(depth_center, depth_roi // 2, h_total - depth_roi // 2))
            z0 = int(np.clip(c - depth_roi // 2, 0, h_total - depth_roi))
            vol, seg = vol[:, z0:z0 + depth_roi, :], seg[:, z0:z0 + depth_roi, :]
        name = f"{prefix}_seg{start}"
        items.append(save_npy_pair(vol, seg, out, name))
        print(f"  {name}: {vol.shape} (D,H,W) fg={float((seg>0).mean()):.4f}")
    make_splits(items, n_folds, out)
    print(f"total volumes: {len(items)}")


def mode_synth(out, n=8, shape=(32, 96, 96)):
    """合成数据：随机背景 + 随机椭圆'汗腺'，仅用于冒烟测试"""
    os.makedirs(out, exist_ok=True)
    items = []
    rng = np.random.RandomState(42)
    for i in range(n):
        vol = (rng.rand(*shape) * 40 + 10).astype(np.uint8)
        seg = np.zeros(shape, np.uint8)
        for _ in range(rng.randint(3, 8)):
            z, y, x = rng.randint(0, shape[0]), rng.randint(40, shape[1] - 40), rng.randint(40, shape[2] - 40)
            rz, ry, rx = rng.randint(2, 6), rng.randint(8, 20), rng.randint(8, 20)
            zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
            m = ((zz - z) / rz) ** 2 + ((yy - y) / ry) ** 2 + ((xx - x) / rx) ** 2 <= 1
            seg[m] = 255
            vol[m] = np.clip(vol[m] + 120, 0, 255)
        items.append(save_npy_pair(vol, seg, out, f"synth{i:03d}"))
    make_splits(items, 4, out)
    print(f"synthetic volumes: {len(items)} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["tif", "bmp", "synth"])
    ap.add_argument("--src", default=None, help="tif/bmp 模式的源目录")
    ap.add_argument("--out", required=True, help="输出 npy 目录")
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--manifest", default=None,
                    help='JSON: {"name": {"img": 路径, "seg": 路径}}，tif 模式可直接指定配对，无需复制原文件')
    ap.add_argument("--depth-roi", type=int, default=288)
    ap.add_argument("--depth-center", type=int, default=140)
    args = ap.parse_args()

    if args.mode == "tif":
        assert args.src or args.manifest, "tif 模式需要 --src 或 --manifest"
        mode_tif(args.src, args.out, args.n_folds, args.manifest)
    elif args.mode == "bmp":
        assert args.src, "bmp 模式需要 --src"
        mode_bmp(args.src, args.out, args.n_folds, args.depth_roi, args.depth_center)
    else:
        mode_synth(args.out)
