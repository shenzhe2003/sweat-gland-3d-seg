"""
评估 —— 滑动窗口推理 + 论文 Table I 五项指标
=================================================
用法：
  python evaluate.py --data-dir data/volumes --ckpt runs/exp1/fold0/best.pt --fold 0
  python evaluate.py --data-dir data/volumes --ckpt runs/exp1/fold{}/best.pt --all-folds  # 4-fold 汇总

指标定义（逐 volume 计算后取均值）：
  Dice = 2TP / (2TP + FP + FN)
  IoU  = TP / (TP + FP + FN)
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  Accuracy  = (TP + TN) / (TP + FP + FN + TN)   # 含背景的逐体素准确率
"""

import os
import json
import argparse
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from model import build_model


def compute_metrics(pred_bin, gt_bin):
    tp = float((pred_bin & (gt_bin > 0)).sum())
    fp = float((pred_bin & (gt_bin == 0)).sum())
    fn = float((~pred_bin & (gt_bin > 0)).sum())
    tn = float((~pred_bin & (gt_bin == 0)).sum())
    eps = 1e-8
    return {
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
    }


@torch.no_grad()
def infer_volume(model, vol, patch, device, overlap=0.5, sw_batch=1, amp=True):
    """滑动窗口推理整个 volume。vol: (D,H,W) uint8 -> pred: (D,H,W) 0/1（裁回原始深度）"""
    model.eval()
    d_orig = vol.shape[0]
    # 深度 pad 到 32 倍数（模型 5 次下采样的硬约束）
    d_pad = int(np.ceil(d_orig / 32) * 32)
    if d_pad > d_orig:
        vol = np.pad(vol, ((0, d_pad - d_orig), (0, 0), (0, 0)), mode="reflect")
    x = torch.from_numpy(vol.astype(np.float32) / 255.0)[None, None].to(device)  # (1,1,D,H,W)
    with torch.amp.autocast("cuda", enabled=amp):
        logits = sliding_window_inference(
            x, roi_size=patch, sw_batch_size=sw_batch, predictor=model,
            overlap=overlap, mode="gaussian", device=device,
        )
    pred = (torch.sigmoid(logits) > 0.5).float()[0, 0].cpu().numpy().astype(bool)
    return pred[:d_orig]   # 裁回原始深度


def evaluate_ckpt(ckpt_path, data_dir, fold, device, overlap=0.5, sw_batch=1):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model"])
    patch = ckpt.get("patch", (64, 288, 288))
    print(f"loaded {ckpt_path} (epoch {ckpt.get('epoch')}, patch {patch})")

    split = json.load(open(os.path.join(data_dir, "splits.json")))
    val_items = [os.path.join(data_dir, os.path.basename(it))
                 for i, it in enumerate(split["items"]) if i in set(split["folds"][fold])]

    rows = []
    for img_path in val_items:
        seg_path = img_path.replace("_img.npy", "_seg.npy")
        vol, gt = np.load(img_path), np.load(seg_path)
        pred = infer_volume(model, vol, patch, device, overlap, sw_batch)
        m = compute_metrics(pred, gt > 0)
        rows.append(m)
        print(f"  {os.path.basename(img_path):40s} dice {m['dice']:.4f}  iou {m['iou']:.4f}  "
              f"prec {m['precision']:.4f}  rec {m['recall']:.4f}  acc {m['accuracy']:.4f}")
    return rows


def fmt(rows):
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in rows]
        out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/volumes")
    ap.add_argument("--ckpt", default="runs/exp1/fold{}/best.pt",
                    help="checkpoint 路径；--all-folds 时用 {} 作 fold 占位符")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--sw-overlap", type=float, default=0.5)
    ap.add_argument("--sw-batch", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.all_folds:
        all_rows = []
        split = json.load(open(os.path.join(args.data_dir, "splits.json")))
        n_actual = len(split["folds"])   # 以 splits.json 实际折数为准
        for f in range(n_actual):
            ck = args.ckpt.replace("{}", str(f))
            if not os.path.exists(ck):
                print(f"[skip] {ck} 不存在")
                continue
            print(f"===== fold {f} =====")
            all_rows += evaluate_ckpt(ck, args.data_dir, f, device, args.sw_overlap, args.sw_batch)
        if all_rows:
            print("\n==== paper Table I 格式（mean ± std）====")
            for k, (m, s) in fmt(all_rows).items():
                print(f"  {k:10s} {m:.4f} ± {s:.4f}")
    else:
        rows = evaluate_ckpt(args.ckpt, args.data_dir, args.fold, device, args.sw_overlap, args.sw_batch)
        print("\n==== mean ± std ====")
        for k, (m, s) in fmt(rows).items():
            print(f"  {k:10s} {m:.4f} ± {s:.4f}")
