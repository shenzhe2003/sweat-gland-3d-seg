"""
训练入口 —— 4 折交叉验证（论文 Sec III-B）
=============================================
用法：
  # 服务器正式训练（以 fold 0 为例）
  python train.py --data-dir data/volumes --fold 0 --out runs/exp1
  # 全部 4 fold
  python train.py --data-dir data/volumes --folds all --out runs/exp1
  # 本机 4GB 冒烟
  python train.py --data-dir data/volumes --smoke
"""

import os
import json
import time
import math
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import parse_args
from model import build_model
from losses import HybridLoss
from data import VolumeDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, warmup_epochs, total_epochs):
    """论文：30 epoch linear warmup 后接 cosine annealing"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_split(data_dir, n_folds):
    with open(os.path.join(data_dir, "splits.json")) as f:
        return json.load(f)


def resolve_items(items, data_dir):
    """splits.json 只存文件名时相对 data_dir 解析（打包到服务器后路径不同也能跑）"""
    return [it if (os.path.isabs(it) and os.path.exists(it))
            else os.path.join(data_dir, os.path.basename(it)) for it in items]


def train_one_fold(cfg, fold, device):
    split = load_split(cfg.data_dir, cfg.n_folds)
    items, folds = split["items"], split["folds"]
    val_idx = set(folds[fold])
    train_items = resolve_items([it for i, it in enumerate(items) if i not in val_idx], cfg.data_dir)
    val_items = resolve_items([it for i, it in enumerate(items) if i in val_idx], cfg.data_dir)
    print(f"[fold {fold}] train volumes: {len(train_items)}, val volumes: {len(val_items)}")

    # 写本 fold 的临时 split（VolumeDataset 读 items 列表）
    fold_dir = os.path.join(cfg.out, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    train_ds = VolumeDataset(cfg.data_dir, patch=cfg.patch, mode="train",
                             subblocks_per_volume=cfg.subblocks_per_volume)
    train_ds.items = train_items
    val_ds = VolumeDataset(cfg.data_dir, patch=cfg.patch, mode="val")
    val_ds.items = val_items
    print(f"[fold {fold}] subblocks/epoch: {len(train_ds)} "
          f"({len(train_items)} volumes x {cfg.subblocks_per_volume})")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.workers)

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    criterion = HybridLoss(cfg.alpha, cfg.beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.warmup_epochs, cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    log_file = os.path.join(fold_dir, "train_log.csv")
    with open(log_file, "w") as f:
        f.write("epoch,lr,train_loss,val_loss,val_dice,time_s\n")

    best_dice = -1.0
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        losses = []
        for step, (img, seg) in enumerate(train_loader):
            if cfg.smoke and step >= 3:      # 冒烟：每 epoch 只跑 3 步
                break
            img, seg = img.to(device), seg.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp):
                logits = model(img)
                loss = criterion(logits, seg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())

        # ---- 验证（中心窗 + Dice）----
        if not losses:
            raise RuntimeError(
                f"本 epoch 没有训练 batch（train subblocks={len(train_ds)} < batch_size={cfg.batch_size}）。"
                "减小 --batch-size，或增大 --subblocks-per-volume")
        val_loss, val_dice = float("nan"), float("nan")
        if (epoch + 1) % cfg.val_every == 0 or epoch == cfg.epochs - 1:
            model.eval()
            vlosses, dices = [], []
            with torch.no_grad():
                for img, seg in val_loader:
                    img, seg = img.to(device), seg.to(device)
                    with torch.amp.autocast("cuda", enabled=cfg.amp):
                        logits = model(img)
                        vlosses.append(criterion(logits, seg).item())
                    pred = (torch.sigmoid(logits) > 0.5).float()
                    inter = (pred * seg).sum()
                    dice = (2 * inter + 1e-6) / (pred.sum() + seg.sum() + 1e-6)
                    dices.append(dice.item())
            val_loss, val_dice = float(np.mean(vlosses)), float(np.mean(dices))

        scheduler.step()
        msg = (f"epoch {epoch+1:3d}/{cfg.epochs}  lr {scheduler.get_last_lr()[0]:.2e}  "
               f"train_loss {np.mean(losses):.4f}  val_loss {val_loss:.4f}  "
               f"val_dice {val_dice:.4f}  ({time.time()-t0:.0f}s)")
        print(msg)
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{scheduler.get_last_lr()[0]:.2e},{np.mean(losses):.4f},"
                    f"{val_loss:.4f},{val_dice:.4f},{time.time()-t0:.0f}\n")

        # 保存 best + last
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "patch": cfg.patch, "val_dice": val_dice},
                   os.path.join(fold_dir, "last.pt"))
        if val_dice == val_dice and val_dice > best_dice:   # 非 NaN 且更优
            best_dice = val_dice
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "patch": cfg.patch, "val_dice": val_dice},
                       os.path.join(fold_dir, "best.pt"))
            print(f"  -> best checkpoint saved (val_dice {best_dice:.4f})")

    print(f"[fold {fold}] done. best val_dice (center-crop) = {best_dice:.4f}")
    return best_dice


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  patch: {cfg.patch}  epochs: {cfg.epochs}  batch: {cfg.batch_size}")
    os.makedirs(cfg.out, exist_ok=True)
    with open(os.path.join(cfg.out, "config.json"), "w") as f:
        json.dump(vars(cfg), f, indent=1)

    if cfg.folds == "all":
        # 折数以 splits.json 实际内容为准（样本少时自动减少，如 3 个样本 -> 3 折）
        with open(os.path.join(cfg.data_dir, "splits.json")) as f:
            n_actual = len(json.load(f)["folds"])
        folds = list(range(n_actual))
    else:
        folds = [int(x) for x in cfg.folds.split(",")]

    results = {}
    for fold in folds:
        results[fold] = train_one_fold(cfg, fold, device)
    print("\n==== summary (center-crop val dice) ====")
    for k, v in results.items():
        print(f"fold {k}: {v:.4f}")
    if len(results) > 1:
        vals = list(results.values())
        print(f"mean ± std: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
