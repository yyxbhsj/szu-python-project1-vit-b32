# project 1 · 用纯 NumPy 实现 OpenCLIP ViT-B/32 图像编码器

本项目用**纯 NumPy**（不依赖 PyTorch）复现 OpenCLIP ViT-B/32 图像编码器的完整前向过程：
输入 `[B, 3, 224, 224]` 的 float32 图像（CLIP 均值/方差归一化后），
输出 `[B, 512]` 的 L2 单位向量。权重由课程提供（152 个张量，约 8785 万参数），**无需训练**。

仓库里唯一的实现文件是 [`image_encoder.py`](image_encoder.py)，它完成了课程 starter 中的六组 TODO：

| TODO | 内容 |
| --- | --- |
| TODO-01 | `patchify`：`[B,3,224,224]` → 49 个 32×32 patch → `[B,49,3072]` |
| TODO-02 | `split_heads`：`[B,50,768]` → `[B,12,50,64]` |
| TODO-03 | 缩放点积注意力（qkv、softmax、head 合并、输出投影） |
| TODO-04 | Transformer block（pre-LN 的两条残差 + MLP/GELU） |
| TODO-05 | 嵌入（patch 线性投影、CLS token、位置编码） |
| TODO-06 | `__call__`（取 CLS 行、`ln_post`、输出投影、L2 单位化） |

## 结果

官方 `check.py` 的八个阶段全部 PASS（本机实测，Python 3.14 / NumPy 2.2.6）：

```
assets       PASS: 152 matching visual tensors and ten image/reference pairs
preprocess   PASS shape=(10, 3, 224, 224) max_abs=0.000e+00 mean_abs=0.000e+00
patches      PASS shape=(2, 49, 3072) max_abs=0.000e+00 mean_abs=0.000e+00
tokens       PASS shape=(2, 50, 768) max_abs=7.153e-06 mean_abs=2.597e-07
block0       PASS shape=(2, 50, 768) max_abs=7.540e-06 mean_abs=2.455e-07
last_tokens  PASS shape=(2, 50, 768) max_abs=4.196e-05 mean_abs=2.438e-06
output       PASS shape=(10, 512) max_abs=7.302e-07 mean_abs=8.244e-08
batch        PASS shape=(3, 512) max_abs=4.992e-07 mean_abs=7.769e-08
Requested numerical checks passed.
```

各阶段误差均比课程容差（内部 `2e-4`、最终输出 `2e-5`）小 1–2 个数量级；
所有中间张量与输出都保持 `float32`，最终输出每一行都是单位向量。

## 目录

```
image_encoder.py              # 本项目的实现（六个 TODO 全部完成，含逐步注释）
requirements.txt              # 依赖列表（与课程 starter 一致）
report/main.tex               # 实验报告 LaTeX 源文件（XeLaTeX 编译）
report/main.pdf               # 实验报告 PDF
report/figs/vit_overview.png  # 报告里的 ViT 整体结构图（模板自带图片）
```

## 环境

```
Python 3.14.0
numpy 2.2.6
Pillow 12.0.0
imageio 2.37.2
```

安装依赖（`requirements.txt` 与课程 starter 一致）：

```bash
pip install -r requirements.txt
```

## 如何复现

模型权重（`weights/image_encoder.npz`，约 326 MB）与参考数据（`assets/`）由课程提供，
体积超过 GitHub 限制，**不随本仓库分发**。复现方式：

1. 取课程的 starter 工程 `project1-numpy-student/`（内含 `check.py`、`run.py`、`helpers.py`、
   `config.json`、`weights/`、`assets/`）；
2. 用本仓库的 `image_encoder.py` 覆盖其中的同名文件（其余文件课程提供的文件不需要改动）；
3. 在工程根目录运行：

```bash
python check.py --stage all
```

约 90–120 秒（大部分时间用于校验 326 MB 权重与 10 张图的推理），预期输出为上面 8 行 PASS。

也可以逐阶段检查（对应上面的 TODO 顺序）：

```bash
python check.py --stage assets      # 权重 / 图片 / 参考数据校验
python check.py --stage preprocess  # 图像预处理（课程提供）
python check.py --stage patches     # TODO-01
python check.py --stage tokens      # TODO-02 / 03 / 05
python check.py --stage block0      # TODO-04
python check.py --stage last_tokens # 12 层完整前向
python check.py --stage output      # TODO-06，全部 10 张图
python check.py --stage batch       # 乱序 + 重复输入（索引 [1,0,1]）
```

导出编码结果：

```bash
python run.py --output output_tokens.npz   # Saved (10, 512)
```

## 实现要点（易错处）

- **权重布局**：注意力的 qkv / 输出投影与 MLP 的线性层权重按 `[out, in]` 存储，需要 `.T`；
  而输出投影 `visual.proj` 已经是 `[in, out] = [768, 512]`，**不要**转置。
- **head 合并**：`context` 的形状是 `[B, 12, 50, 64]`，必须先用 `transpose(0,2,1,3)`
  再 `reshape(..., 50, 768)`，否则特征会按 head 交错，结果错误。
- **float32 契约**：`np.sqrt(head_dim)` 会返回 `np.float64` 标量，用它做除法会把整个数组
  静默提升为 float64，`check.py` 会报 `block0: use float32`；需显式转成 `np.float32`。
- **CLS 与位置编码**：CLS 在 token 序列的第 0 行；patch 行序为「先行后列」，
  必须与位置编码的第 1–49 行一一对应。

## 报告

实验报告见 [`report/main.pdf`](report/main.pdf)（LaTeX 源文件 `report/main.tex`，用 XeLaTeX 编译）。
