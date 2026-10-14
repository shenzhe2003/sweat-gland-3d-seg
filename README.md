# 3D 汗腺分割复现（SweatGland-3D-Seg）

紧贴论文 *3D Deep-learning-based Segmentation of Human Skin Sweat Glands and Their 3D Morphological Response to Temperature Variations*（Pei et al.）的**汗腺分割部分**复现代码。

**仓库同步链路**：GitHub（主仓库）→ Gitee（镜像，服务器只通 Gitee）→ 服务器 `git clone`。数据文件太大不入库，获取方式见下文「数据获取」。

## 论文 ↔ 代码对应表

| 论文 | 代码 | 说明 |
|---|---|---|
| Fig.1 整体架构 | `model.py: SweatGlandNet` | 完整装配 |
| Swin-UNETR 骨干（patch 2×2×2, embed 24, 5 级） | `model.py: SwinEncoder` | MONAI 官方 `BasicLayer`/`PatchMergingV2`/`SwinTransformerBlock` |
| ECA 跳连注意力（Sec II-B） | `model.py: ECAAttention` | GAP→1D conv(k=3)→sigmoid |
| 深层 DLK Block + DFF（Sec II-C, [35]=D-Net） | `model.py: DLKBlock / DFF` | 严格按 D-Net 论文公式 |
| 浅层 Res Block + 通道拼接 | `model.py: ResidualBlock` | |
| Salience Layer | `model.py: SalienceLayer` | Conv Block + Channel Mixer（D-Net Sec 3.3） |
| 混合损失 0.9·BCE + 0.1·Dice（Sec II-D） | `losses.py` | |
| 288×288×64 subblock（Sec III-A） | `data.py` | 深度 pad 到 32 倍数（模型硬约束） |
| AdamW / 300ep / lr 1e-5 / 30ep warmup / cosine / 4-fold（Sec III-B） | `train.py` | |
| 滑动窗口推理（摘要） | `evaluate.py` | MONAI `sliding_window_inference` |
| Table I 五项指标 | `evaluate.py: compute_metrics` | Dice/IoU/Precision/Recall/Accuracy |

模型参数量：**7,752,166**（论文 7,375,884，+5.1%，差异来源见“推断点”）。

## 环境安装

```bash
pip install -r requirements.txt   # torch/monai/numpy/tifffile/pillow
```

- 论文环境：Python 3.8.8 / PyTorch 1.13.1 / MONAI 1.3.0
- 本复现验证环境：Python 3.13 / PyTorch 2.13 / MONAI 1.6.1（API 兼容）
- 训练需要 NVIDIA GPU：论文用 2×A6000 48G；**RTX 2050 4GB 实测只能跑 ≤(64,160,160)**，论文 patch (64,288,288) 需 ~12GB，请上服务器

## 数据准备

**数据不在 git 仓库里**（3 个 800³ volume 解压前约 2.9GB，超出合理仓库体积），
需按下述方式之一准备到 `data/volumes/`：

**方式 A：天池原始 tif 生成（推荐，可无限扩容）**

从天池数据集页下载（<https://tianchi.aliyun.com/dataset/195640>，IAI-Lab）：
Fig.4-5 的 3 对 volume+mask（P0004/P0013/P0019，各 800×800×800）。
把 `data/manifest_3doct.json` 里的路径改成本地实际路径后：

```bash
python prepare_data.py --mode tif --manifest data/manifest_3doct.json --out data/volumes
```

**方式 B：直接用已转好的 npy 包**

`sweat_gland_3d_seg_server.zip`（约 0.61GB，含 `data/volumes` 全部 npy）解压到仓库根目录即可。

**样本说明**（经内容哈希查重，已剔除 Fig.3/Fig.8 中与 P0019 重复的副本；mask 已验证为人工标注）：

| 样本 | volume | GT | 前景占比 |
|---|---|---|---|
| P0004 | 800×800×800 | mask.tif | 0.79% |
| P0013 | 800×800×800 | mask.tif | 0.64% |
| P0019 | 800×800×800 | mask.tif（=Fig.3 annotation=Fig.8 Label） | 0.74% |

每折划分（留一法，论文为 4-fold，样本不足时自动降级）：

| Fold | 训练 | 测试 |
|---|---|---|
| fold 0 | P0004 + P0013 | P0019 |
| fold 1 | P0004 + P0019 | P0013 |
| fold 2 | P0013 + P0019 | P0004 |

## 快速开始

```bash
# 0. 一键冒烟（本机 4GB，约 2 分钟，验证全链路）
python smoke_test.py

# 1. 正式训练（服务器；fold 0 为例，默认 patch=64,288,288=论文原版）
python train.py --data-dir data/volumes --folds 0 --out runs/exp1

# 2. 全部 4 折交叉验证
python train.py --data-dir data/volumes --folds all --out runs/exp1

# 3. 评估（滑动窗口推理 + Table I 指标；4 折汇总 mean±std）
python evaluate.py --data-dir data/volumes --ckpt "runs/exp1/fold{}/best.pt" --all-folds
```

常用参数：`--patch 64,288,288`（必须是 32 的倍数）、`--batch-size 4`、`--epochs 300`、
`--lr 1e-5`、`--sw-overlap 0.5`、`--no-amp`。本机 4GB 示例：
`python train.py --data-dir data/volumes_bmp --patch 32,192,192 --batch-size 1`。

⚠️ **小样本必读**：volume 少时（如本地 3 个），每个 epoch 的 subblock 数 =
训练 volume 数 × `--subblocks-per-volume`（默认 10，对齐论文 1567 subblocks / ~150 volume）。
若小于 `--batch-size` 会直接报错提示——减小 batch 或增大 `--subblocks-per-volume`即可。

## 论文预期结果（Table I）

| 指标 | 论文值 | 说明 |
|---|---|---|
| Dice | 0.8925 ± 0.0355 | 4-fold CV |
| IoU | 0.8062 ± 0.0450 | |
| Precision | 0.8381 ± 0.0692 | |
| Recall | 0.9496 ± 0.0912 | |
| Accuracy | 0.9955 ± 0.0024 | |

**关于指标预期**：深度上下文已是论文原版（800 帧 volume、288×288×64 subblock），
但本地只有 3 个训练样本（论文用约 150 个 volume / 1567 个 subblock），3 折留一验证
的样本量远小于论文 4-fold，Dice 等指标**大概率低于论文值**；差距主要来自数据量而非
代码实现。后续从天池补充 volume+GT 配对（见 manifest 用法）后重训即可逼近。

## 推断点（论文未明确、本复现的工程选择）

1. **subblock 裁剪策略**：论文未描述。汗腺前景仅 ~0.5%，纯随机裁窗几乎全背景，
   故采用“以前景色素为中心随机抖动裁窗”（`data.py: fg_bias`）。
2. **归一化**：论文未指定，直接用 x/255。
3. **数据增强**：论文 Fig.1 提到 Data Augmentation 但未列明细；`data.py: _augment`
   实现随机翻转/90°旋转/亮度抖动（train 模式默认开启）。
4. **decoder 深浅层划分**：论文只说“浅层 Res、深层 DLK+DFF”，未给边界；
   本复现以通道数划分：≥96 通道用 DLK+DFF，≤48 用 Res+concat。
5. **ECA kernel size**：取 ECA-Net 默认 k=3。
6. **滑窗 overlap**：论文未给，取 0.5。
7. **参数量 +5%**：来自 Salience Layer / head 等实现细节差异（如 head_conv 用了
   ResidualBlock），架构组件与论文 Fig.1 一一对应。
8. **Dice loss 形式**：MONAI `DiceLoss(sigmoid=True)` 默认配置（soft Dice）。
9. **subblock 数量**：论文预裁 1567 个 subblock（~150 volume）；本复现在线随机裁，
   每 epoch 每 volume 采 `--subblocks-per-volume`（默认 10）个，总量可调。
10. **tGT-OCT 增强**：论文用 tGT-OCT [39] 增强原图；P0019 用的是 Fig.3 的 denoise
    版本（即增强后），P0004/P0013 是否为增强版论文未标注，按原样使用。

## GitHub → Gitee → 服务器 工作流

```bash
# ---- 1. 本机：提交并推送到 GitHub（主仓库）----
git remote add origin https://github.com/shenzhe2003/sweat-gland-3d-seg.git
git push -u origin main

# ---- 2. Gitee 镜像（二选一）----
# 方式 a：Gitee 网页端「+ → 从 GitHub 导入」或 仓库管理 → 仓库镜像管理 → 同步
# 方式 b：本机直接推一份到 Gitee（<你的用户名> 换成你的 Gitee 账号）
git remote add gitee https://gitee.com/<你的用户名>/sweat-gland-3d-seg.git
git push gitee main

# ---- 3. 服务器（只通 Gitee）：拉取 + 数据 + 训练 ----
git clone https://gitee.com/<你的用户名>/sweat-gland-3d-seg.git
cd sweat-gland-3d-seg
pip install -r requirements.txt
# 数据：解压 sweat_gland_3d_seg_server.zip 到仓库根目录（得到 data/volumes/），
#      或按「数据准备」方式 A 从天池 tif 生成
python train.py --data-dir data/volumes --folds all --out runs/exp1 --batch-size 4
python evaluate.py --data-dir data/volumes --ckpt "runs/exp1/fold{}/best.pt" --all-folds
```

训练日志：`runs/exp1/foldN/train_log.csv`（loss/lr/val_dice 逐 epoch）；
checkpoint：`best.pt`（按 val dice）/ `last.pt`。`runs/` 与 `*.npy` 均已在 `.gitignore` 中，不会误提交。

后续拿到天池更多 volume+GT 配对：往 `data/manifest_3doct.json` 加条目 → 重跑
`prepare_data.py --mode tif --manifest ...` → `git push` 三个端同步更新。

## 文件结构（git 仓库内）

```
sweat-gland-3d-seg/
├── model.py          # 网络（ECA/DLK/DFF/Salience/SwinEncoder/SweatGlandNet）
├── losses.py         # 0.9 BCE + 0.1 Dice
├── data.py           # volume 数据集 + subblock 采样
├── prepare_data.py   # tif(manifest)/BMP/合成 -> npy + splits
├── train.py          # 训练（N-fold CV, warmup+cosine, AMP）
├── evaluate.py       # 滑动窗口推理 + 五项指标
├── config.py         # 论文超参集中管理
├── smoke_test.py     # 一键冒烟
├── requirements.txt
├── .gitignore        # 已排除 runs/ *.npy data/volumes/ 等大文件
└── data/
    └── manifest_3doct.json   # 天池 tif -> npy 的配对清单（路径需按本地实际修改）
```
