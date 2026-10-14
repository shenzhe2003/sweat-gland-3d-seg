"""
训练配置 —— 论文 Sec III-B 全部超参集中于此
=============================================
论文值：batch 4 / AdamW wd 1e-6 / 300 epochs / lr 1e-5 /
        30 epoch warmup + cosine annealing / 4-fold CV (train:val=3:1) /
        loss = 0.9 BCE + 0.1 Dice / subblock 288x288x64
本机 RTX 2050 4GB 冒烟：--smoke 一键切换小配置
"""

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="3D Sweat Gland Segmentation (paper reproduction)")
    # ---- 数据 ----
    p.add_argument("--data-dir", default="data/volumes", help="prepare_data.py 输出的 npy 目录")
    p.add_argument("--patch", default="64,288,288", help="subblock (D,H,W)，论文 64,288,288")
    p.add_argument("--subblocks-per-volume", type=int, default=10,
                   help="每 epoch 每个 volume 采多少 subblock（论文 1567/~150 volume≈10；volume 少时防空 loader）")
    # ---- 训练（论文 Sec III-B）----
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.9, help="BCE 权重")
    p.add_argument("--beta", type=float, default=0.1, help="Dice 权重")
    # ---- 交叉验证 ----
    p.add_argument("--folds", default="all", help="'all' 或逗号分隔的 fold 列表，如 0,1")
    p.add_argument("--n-folds", type=int, default=4)
    # ---- 推理/评估 ----
    p.add_argument("--sw-overlap", type=float, default=0.5, help="滑动窗口重叠率（论文未指定，取常用值）")
    p.add_argument("--sw-batch", type=int, default=1, help="滑窗 batch（显存受限时调小）")
    # ---- 工程 ----
    p.add_argument("--amp", action="store_true", default=True, help="混合精度（默认开）")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--val-every", type=int, default=5, help="每多少 epoch 验证一次")
    p.add_argument("--out", default="runs/exp1", help="输出目录")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="冒烟测试：小 patch、2 epoch、少量 step")
    args = p.parse_args()

    args.patch = tuple(int(x) for x in args.patch.split(","))
    for d in args.patch:
        if d % 32 != 0:
            raise ValueError(f"patch 每个维度必须是 32 的倍数（模型 5 次下采样的硬约束），当前: {args.patch}")
    if args.smoke:
        args.patch = (32, 96, 96)
        args.epochs = 2
        args.batch_size = 1
        args.val_every = 1
        args.out = "runs/smoke"
    return args
