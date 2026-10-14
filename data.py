"""
数据集与 3D subblock 采样
==========================
论文 Sec III-A：从 800x800x800 volume 裁 288x288x64 (X x Z x Y) subblock 训练，
共 1567 个 subblock。论文未描述裁剪策略——汗腺前景仅占 ~0.5%，
纯随机裁窗几乎全是背景，因此采用「以前景色素为中心随机抖动裁窗」，
保证每个训练 subblock 含汗腺（工程推断，README 已注明）。
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

# 归一化：OCT 原图 uint8 [0,255]，直接 /255（论文未指定，README 已注明）
def normalize(img):
    return img.astype(np.float32) / 255.0


class VolumeDataset(Dataset):
    """
    读取 prepare_data.py 生成的 npy volume：
      <data_dir>/*_img.npy  -> (D, H, W) uint8，D=B-scan 帧序号, H=轴向深度, W=横向
      <data_dir>/*_seg.npy  -> (D, H, W) uint8 (0/1)
    每个样本按 patch (pd, ph, pw) 随机裁一个 subblock。
    """
    def __init__(self, data_dir, patch=(64, 288, 288), split_file=None, mode="train",
                 fg_bias=True, subblocks_per_volume=10):
        self.patch = patch
        self.mode = mode
        self.fg_bias = fg_bias and (mode == "train")
        # train 模式每 epoch 对每个 volume 重复采多个 subblock（论文：1567 subblocks / ~150 volume ≈ 10）；
        # 避免 volume 太少时 batch_size=4 + drop_last 导致空 loader（模型不训练）
        self.repeats = max(1, subblocks_per_volume) if mode == "train" else 1
        self.items = self._load_items(data_dir, split_file)

    def _load_items(self, data_dir, split_file):
        if split_file and os.path.exists(split_file):
            import json
            with open(split_file) as f:
                items = json.load(f)["items"]
            # 统一相对 data_dir 解析：splits.json 只存文件名，换机器/换目录都能跑
            return [it if (os.path.isabs(it) and os.path.exists(it))
                    else os.path.join(data_dir, os.path.basename(it)) for it in items]
        # 无 split 文件时用全部
        return sorted(glob.glob(os.path.join(data_dir, "*_img.npy")))

    def __len__(self):
        return len(self.items) * self.repeats

    def _pad_depth(self, vol, seg):
        """深度方向 pad 到 32 的倍数（模型 5 次 2 倍下采样的硬约束；本机 16 帧段 -> pad 到 32）"""
        d = vol.shape[0]
        target = int(np.ceil(d / 32) * 32)
        if target > d:
            vol = np.pad(vol, ((0, target - d), (0, 0), (0, 0)), mode="reflect")
            seg = np.pad(seg, ((0, target - d), (0, 0), (0, 0)), mode="reflect")
        return vol, seg

    def _crop(self, vol, seg):
        vol, seg = self._pad_depth(vol, seg)
        pd, ph, pw = self.patch
        d, h, w = vol.shape
        # 深度/高/宽任一小于 patch 时 pad 到 patch（本机 16 帧段深度刚好为 16 < 64 时由配置调整 patch）
        pad = [(0, max(0, pd - d)), (0, max(0, ph - h)), (0, max(0, pw - w))]
        if any(p[1] > 0 for p in pad):
            vol = np.pad(vol, pad, mode="constant")
            seg = np.pad(seg, pad, mode="constant")
            d, h, w = vol.shape

        if self.mode == "train":
            if self.fg_bias and seg.sum() > 0:
                # 以前景色素为中心随机裁窗（保证 batch 内有正样本）
                fg = np.argwhere(seg > 0)
                c = fg[np.random.randint(len(fg))]
                z0 = int(np.clip(c[0] - pd // 2 + np.random.randint(-pd // 4, pd // 4 + 1), 0, d - pd))
                y0 = int(np.clip(c[1] - ph // 2 + np.random.randint(-ph // 4, ph // 4 + 1), 0, h - ph))
                x0 = int(np.clip(c[2] - pw // 2 + np.random.randint(-pw // 4, pw // 4 + 1), 0, w - pw))
            else:
                z0 = np.random.randint(0, d - pd + 1)
                y0 = np.random.randint(0, h - ph + 1)
                x0 = np.random.randint(0, w - pw + 1)
        else:
            # val/test：中心裁窗（配合滑动窗口推理脚本做全 volume 评估）
            z0, y0, x0 = (d - pd) // 2, (h - ph) // 2, (w - pw) // 2

        return vol[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw], seg[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]

    def _augment(self, v, s):
        """train 模式 3D 增强（论文 Fig.1 Salience Layer 图标含 Data Augmentation，论文未列明细）：
        随机翻转 / H-W 平面随机 90° 旋转 / 亮度抖动。作用于归一化后的张量。"""
        v, s = v.copy(), s.copy()
        for ax in (0, 1, 2):                                   # 三轴随机翻转
            if np.random.rand() < 0.5:
                v, s = np.flip(v, ax), np.flip(s, ax)
        if np.random.rand() < 0.5:                             # H-W 平面随机旋转 90/180/270°
            k = np.random.randint(1, 4)
            v, s = np.rot90(v, k, axes=(1, 2)), np.rot90(s, k, axes=(1, 2))
        if np.random.rand() < 0.5:                             # 亮度抖动
            v = v * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05)
        return np.ascontiguousarray(v), np.ascontiguousarray(s)

    def __getitem__(self, idx):
        img_path = self.items[idx % len(self.items)]   # repeats：同 volume 每 epoch 采多个不同 subblock
        seg_path = img_path.replace("_img.npy", "_seg.npy")
        vol = np.load(img_path)
        seg = np.load(seg_path)
        v, s = self._crop(vol, seg)
        v, s = normalize(v), s.astype(np.float32)
        if self.mode == "train":
            v, s = self._augment(v, s)
        return torch.from_numpy(v[None]), torch.from_numpy(s[None])   # (1,D,H,W)
