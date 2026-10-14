"""
损失函数 —— 论文 Sec II-D
  L = alpha * BCE + beta * Dice,  alpha=0.9, beta=0.1
BCE 为主（解决前景/背景极度不均衡），Dice 精修边界。
"""

import torch
import torch.nn as nn
from monai.losses import DiceLoss


class HybridLoss(nn.Module):
    def __init__(self, alpha=0.9, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()
        # sigmoid=True：直接吃 logits；squared_pred/batch 保持 MONAI 默认（论文未特别说明）
        self.dice = DiceLoss(sigmoid=True, to_onehot_y=False)

    def forward(self, logits, target):
        target = target.float()
        return self.alpha * self.bce(logits, target) + self.beta * self.dice(logits, target)
