"""
网络模型 —— 严格对应论文 Fig.1
====================================

论文：3D Deep-learning-based Segmentation of Human Skin Sweat Glands
骨干：Swin-UNETR 式 5 级 Swin Transformer encoder（patch 2x2x2, embed dim 24）
改进（ Fig.1 标注的三处 ）：
  1. Salience Layer      —— 输入端的 Conv Block + Channel Mixer（结构取自 D-Net, arXiv 2403.10674）
  2. ECA                 —— 每级 skip connection 上的高效通道注意力（ECA-Net, CVPR 2020）
  3. 异构 Decoder        —— 深层用 DLK Block + DFF（D-Net 模块），浅层用 Residual Block + 通道拼接

张量布局：PyTorch 3D 卷积标准 (B, C, D, H, W)。
论文写 288x288x64 (X x Z x Y)，其中 X=横向、Z=深度、Y=B-scan 帧序号，
对应到本代码 D=64(帧), H=288(横), W=288(深)。
"""

import torch
import torch.nn as nn

# MONAI 1.6.1 起 SwinTransformerBlock 位于 nets.swin_unetr 内（1.3.0 时位于 networks.blocks）
try:
    from monai.networks.blocks import SwinTransformerBlock, PatchMergingV2, BasicLayer
except ImportError:
    from monai.networks.nets.swin_unetr import SwinTransformerBlock, PatchMergingV2, BasicLayer

# ============================================================
# 基础构件
# ============================================================

class LayerNorm3d(nn.Module):
    """channel-first LayerNorm（等价 GroupNorm(1, C)，与 MONAI Swin block 内部一致）"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        return self.norm(x)


class MLP3d(nn.Module):
    """1x1x1 conv -> GELU -> 1x1x1 conv（MONAI Swin block 同款 MLP）"""
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Conv3d(dim, hidden, 1)
        self.fc2 = nn.Conv3d(hidden, dim, 1)

    def forward(self, x):
        return self.fc2(nn.functional.gelu(self.fc1(x)))


class ResidualBlock(nn.Module):
    """decoder 浅层用的残差卷积块（论文 Sec II-C：浅层特征简单，用 Res-block 做通道拼接）"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_c), out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_c), out_c)
        self.shortcut = nn.Conv3d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        h = nn.functional.leaky_relu(self.norm1(self.conv1(x)), 0.01)
        h = self.norm2(self.conv2(h))
        return nn.functional.leaky_relu(h + self.shortcut(x), 0.01)


# ============================================================
# ECA：Efficient Channel Attention（论文 Sec II-B, ECA-Net CVPR2020）
# ============================================================

class ECAAttention(nn.Module):
    """
    GAP -> 跨通道 1D 卷积(k=3) -> Sigmoid，不做维度缩减。
    挂在 encoder 每一级输出的 skip connection 上（Fig.1 中 5 个 ECA）。
    """
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.avg_pool(x).view(b, 1, c)      # (B,1,C) 通道描述符
        y = self.conv(y)                         # 跨通道 1D 卷积
        y = torch.sigmoid(y).view(b, c, 1, 1, 1)
        return x * y


# ============================================================
# DLK：Dynamic Large Kernel（D-Net Sec 3.1, arXiv 2403.10674）
# ============================================================

class DLKCore(nn.Module):
    """
    DLK 本体（D-Net 论文公式 1-5）：
      1x1 conv 降维 C->C/2
      级联 DWConv(5x5x5, d=1) -> DWConv(7x7x7, d=3)，两路 concat 回 C
        （级联而非并行，等效感受野 23x23x23）
      空间动态选择：通道均值/最大池化 -> 7x7x7 conv -> sigmoid -> 两路加权相加
      通道校准：GAP -> 1x1 conv -> sigmoid -> 加权
      残差：out = w_ch * x_ch + x_in
    """
    def __init__(self, dim):
        super().__init__()
        self.project = nn.Conv3d(dim, dim // 2, 1)
        self.dw1 = nn.Conv3d(dim // 2, dim // 2, 5, padding=2, groups=dim // 2)          # (5, d=1)
        self.dw2 = nn.Conv3d(dim // 2, dim // 2, 7, padding=9, dilation=3, groups=dim // 2)  # (7, d=3)
        self.spatial_conv = nn.Conv3d(2, 2, 7, padding=3)   # Conv7 on [AVP;MAP]
        self.ch_conv = nn.Conv3d(dim, dim, 1)               # Conv1 on AVGPool

    def forward(self, x):
        x_proj = self.project(x)                              # C/2
        x1 = self.dw1(x_proj)                                 # 分支1: (5,1)
        x2 = self.dw2(x1)                                     # 分支2: (7,3) 级联在 x1 上
        x_sp = torch.cat([x1, x2], dim=1)                     # -> C
        # 空间动态选择
        w_avp = x_sp.mean(dim=1, keepdim=True)                # 通道均值池化
        w_map = x_sp.amax(dim=1, keepdim=True)                # 通道最大池化
        w = torch.sigmoid(self.spatial_conv(torch.cat([w_avp, w_map], dim=1)))
        x_ch = w[:, 0:1] * x_sp + w[:, 1:2] * x_sp
        # 通道校准
        w_ch = torch.sigmoid(self.ch_conv(x_ch.mean(dim=(2, 3, 4), keepdim=True)))
        return w_ch * x_ch + x                                # 残差（对 core 输入）


class DLKModule(nn.Module):
    """D-Net 论文 DLK Module 封装：1x1 conv -> GELU -> DLK core -> 1x1 conv -> 残差"""
    def __init__(self, dim):
        super().__init__()
        self.conv_in = nn.Conv3d(dim, dim, 1)
        self.core = DLKCore(dim)
        self.conv_out = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        return self.conv_out(self.core(nn.functional.gelu(self.conv_in(x)))) + x


class DLKBlock(nn.Module):
    """
    D-Net 论文 DLK Block（替代 Swin 的 self-attention 位置）：
    LN -> DLK Module -> 残差 -> LN -> MLP -> 残差
    对应 Fig.1 右上角 "Dynamic Large Kernel Block" 图标。
    """
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = LayerNorm3d(dim)
        self.dlkm = DLKModule(dim)
        self.norm2 = LayerNorm3d(dim)
        self.mlp = MLP3d(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.dlkm(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# DFF：Dynamic Feature Fusion（D-Net Sec 3.2）
# ============================================================

class DFF(nn.Module):
    """
    融合两路同分辨率特征 F1, F2（decoder 上采样特征 与 encoder skip 特征）：
      通道分支：concat(F1,F2) -> GAP -> 1x1 conv -> sigmoid 得通道权重 w_ch，
                w_ch * concat -> 1x1 conv 压回 C
      空间分支：1x1 conv(F1) + 1x1 conv(F2) -> sigmoid 得空间权重 w_sp
      out = w_sp * F_ch
    """
    def __init__(self, dim):
        super().__init__()
        self.ch_conv = nn.Conv3d(2 * dim, 2 * dim, 1)   # 对 GAP 后的描述符
        self.fuse = nn.Conv3d(2 * dim, dim, 1)          # 通道压回 C
        self.sp_conv1 = nn.Conv3d(dim, dim, 1)
        self.sp_conv2 = nn.Conv3d(dim, dim, 1)

    def forward(self, f1, f2):
        f = torch.cat([f1, f2], dim=1)                              # 2C
        w_ch = torch.sigmoid(self.ch_conv(f.mean(dim=(2, 3, 4), keepdim=True)))
        f_ch = self.fuse(w_ch * f)                                  # -> C
        w_sp = torch.sigmoid(self.sp_conv1(f1) + self.sp_conv2(f2))
        return w_sp * f_ch


# ============================================================
# Salience Layer（D-Net Sec 3.3：Conv Block + Channel Mixer）
# ============================================================

class SalienceLayer(nn.Module):
    """
    输入端显著性层（Fig.1 左下角）：
      Channel Projection Block: 3x3x3 conv(1->C) + BN + LeakyReLU（保持全分辨率）
      Channel Mixer: BN -> 1x1 conv(C->4C) -> GELU -> 3x3 DWConv -> 1x1 conv(4C->C) -> Dropout + 残差
    """
    def __init__(self, in_ch, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_ch, dim, 3, padding=1),
            nn.BatchNorm3d(dim),
            nn.LeakyReLU(0.01),
        )
        self.norm = nn.BatchNorm3d(dim)
        self.fc1 = nn.Conv3d(dim, dim * 4, 1)
        self.dw = nn.Conv3d(dim * 4, dim * 4, 3, padding=1, groups=dim * 4)
        self.fc2 = nn.Conv3d(dim * 4, dim, 1)
        self.drop = nn.Dropout3d(0.1)

    def forward(self, x):
        x = self.conv_block(x)
        shortcut = x
        h = self.norm(x)
        h = nn.functional.gelu(self.fc1(h))
        h = self.dw(h)
        h = self.drop(self.fc2(h))
        return h + shortcut


# ============================================================
# Swin Transformer Encoder（Fig.1 中部，5 级输出 24/48/96/192/384）
# ============================================================

class SwinEncoder(nn.Module):
    """
    patch partition: 2x2x2 stride2 conv（论文 Sec II-A）
    Stage1-4: MONAI 官方 PatchMergingV2（通道翻倍，与 SwinUNETR 一致）
              + 每级 2 个 Swin Transformer Block（BasicLayer 封装，W-MSA/SW-MSA 交替）
    输出 5 级特征（skip0..skip4），通道 24/48/96/192/384，分辨率 H/2..H/32。
    组件全部来自 MONAI 官方实现（论文 baseline SwinUNETR 同源）。
    """
    def __init__(self, embed_dim=24, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                 window_size=(7, 7, 7)):
        super().__init__()
        dims = [embed_dim * (2 ** i) for i in range(5)]   # 24/48/96/192/384

        # Stage0: patch partition（2x2x2）
        self.patch_embed = nn.Conv3d(embed_dim, embed_dim, kernel_size=2, stride=2)

        # Stage1-4: PatchMergingV2 要 channel-last，BasicLayer 要 channel-first，forward 里转置
        self.merge = nn.ModuleList([PatchMergingV2(dims[i], norm_layer=nn.LayerNorm) for i in range(4)])
        self.layers = nn.ModuleList([
            BasicLayer(dim=dims[i + 1], depth=depths[i], num_heads=num_heads[i],
                       window_size=window_size, drop_path=[0.0] * depths[i])
            for i in range(4)
        ])

    def forward(self, s):
        """s: Salience Layer 输出 (B, embed_dim, H, W, D)。返回 5 级特征列表"""
        feats = []
        x = self.patch_embed(s)          # skip0: 24 @ H/2
        feats.append(x)
        for merge, layer in zip(self.merge, self.layers):
            x = x.permute(0, 2, 3, 4, 1).contiguous()   # (B,D,H,W,C) 供 PatchMergingV2
            x = merge(x)
            x = x.permute(0, 4, 1, 2, 3).contiguous()   # (B,C,D,H,W) 供 BasicLayer
            x = layer(x)
            feats.append(x)              # skip1..4
        return feats                     # [24@H/2, 48@H/4, 96@H/8, 192@H/16, 384@H/32]


# ============================================================
# 完整网络
# ============================================================

class SweatGlandNet(nn.Module):
    """
    Fig.1 完整装配：
      Salience -> Encoder(5级) -> 每级 ECA -> Decoder(深层DLK+DFF / 浅层Res+concat)
      -> 与 Salience 特征 DFF 融合 -> conv block -> 1x1 conv -> logits
    """
    def __init__(self, in_ch=1, out_ch=1, embed_dim=24,
                 depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), window_size=(7, 7, 7)):
        super().__init__()
        dims = [embed_dim * (2 ** i) for i in range(5)]
        self.salience = SalienceLayer(in_ch, embed_dim)
        self.encoder = SwinEncoder(embed_dim, depths, num_heads, window_size)

        # 每级 skip connection 上的 ECA
        self.eca = nn.ModuleList([ECAAttention(c) for c in dims])

        # ---- Decoder ----
        # 深层（384->192->96）：upsample + DFF 融合 + DLK Block
        self.up4 = nn.ConvTranspose3d(dims[4], dims[3], kernel_size=2, stride=2)
        self.dff4 = DFF(dims[3])
        self.dlk4 = DLKBlock(dims[3])
        self.up3 = nn.ConvTranspose3d(dims[3], dims[2], kernel_size=2, stride=2)
        self.dff3 = DFF(dims[2])
        self.dlk3 = DLKBlock(dims[2])
        # 浅层（96->48->24）：upsample + 通道 concat + Residual Block
        self.up2 = nn.ConvTranspose3d(dims[2], dims[1], kernel_size=2, stride=2)
        self.res2 = ResidualBlock(dims[1] * 2, dims[1])
        self.up1 = nn.ConvTranspose3d(dims[1], dims[0], kernel_size=2, stride=2)
        self.res1 = ResidualBlock(dims[0] * 2, dims[0])
        # 输出级：上采样回全分辨率，与 Salience 特征 DFF 融合
        self.up0 = nn.ConvTranspose3d(dims[0], dims[0], kernel_size=2, stride=2)
        self.dff0 = DFF(dims[0])
        self.head_conv = ResidualBlock(dims[0], dims[0])
        self.head = nn.Conv3d(dims[0], out_ch, kernel_size=1)

    def forward(self, x):
        s = self.salience(x)                 # 全分辨率 C 通道
        e0, e1, e2, e3, e4 = self.encoder(s)

        # 深层 decoder
        d = self.up4(e4)
        d = self.dff4(d, self.eca[3](e3))
        d = self.dlk4(d)
        d = self.up3(d)
        d = self.dff3(d, self.eca[2](e2))
        d = self.dlk3(d)
        # 浅层 decoder
        d = self.up2(d)
        d = self.res2(torch.cat([d, self.eca[1](e1)], dim=1))
        d = self.up1(d)
        d = self.res1(torch.cat([d, self.eca[0](e0)], dim=1))
        # 输出级
        d = self.up0(d)
        d = self.dff0(d, s)
        d = self.head_conv(d)
        return self.head(d)                  # logits, (B,1,H,W,D)


def build_model(in_ch=1, out_ch=1, embed_dim=24):
    """工厂函数：构建论文模型并返回"""
    return SweatGlandNet(in_ch, out_ch, embed_dim)


if __name__ == "__main__":
    # 参数量自检：论文 Table I 为 7,375,884
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}  (paper: 7,375,884)")
