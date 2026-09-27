# 河道图像分割与堵江识别实验工程

本项目面向固定摄像头河道图像，研究白天/夜间河体语义分割、河道几何变化特征、堵江状态识别和事件告警。工程保留原有研究代码，同时提供统一的 `riverlab` 命令行，用于数据检查、无泄漏划分、训练、评估、消融实验和结果导出。

项目计划形成 EI 论文，并为后续专利和软件著作权材料保留代码、配置、数据版本与实验来源。原始数据、模型权重和运行结果属于本地研究资产，默认不提交 Git。

## 研究流程

```text
白天/夜间图像 + 河体 mask
            │
            ▼
     河体语义分割模型
            │
            ▼
  面积、上下边界及变化特征
            │
            ▼
       堵江状态分类
            │
            ▼
   连续事件回放与告警评价
```

当前优先完成分割模型对比和数据增广/损失函数消融；分类实验在图像对清单和特征版本冻结后运行。

## 数据集

| 数据 | 本地目录 | 作用 | 当前规模 |
| --- | --- | --- | ---: |
| 白天分割裁剪数据 | `segment/VOCdevkit/VOC2007` | 白天河体分割训练及 ROI 重建 | 32,270 块，来自 3,227 张原图 |
| 夜间分割裁剪数据 | `segment/VOCdevkit/VOC2023` | 历史夜间裁剪分割数据 | 22,700 块 |
| 夜间整图数据 | `segment/VOCdevkit/Full_size` | 夜间分割主实验 | 722 张整图及 mask |
| 白天堵江图像对 | 仓库外 `norm-dis-data/data_pairs_normal2dis` | 正常到堵塞的图像对 | 103 个正例目录，另有正常对目录 |
| 夜间堵江图像对 | 仓库外 `norm-dis-data/Nightdata_pairs_normal2dis` | 夜间正常/堵塞图像对 | 7 个正例、47 个正常对 |
| 历史分类特征 | `Classifier_ACC0.93/cnn` | 面积、上下边界和标签的历史 CSV | 243 对 |

白天 mask 存在不同颜色编码，读取时通过 `configs/day_mask_mapping.json` 显式转换为统一类别。内部 mask 约定为 `0=背景、1=河体、255=忽略区域`。

数据目录被 `.gitignore` 排除。克隆仓库后，需要将数据单独复制到以上相对路径，或修改本地配置指向实际数据位置。

## 目录结构

```text
river_warning/
├─ riverlab/                 # 统一实验包和 python -m riverlab 入口
├─ configs/                  # 数据、训练、质量控制和实验矩阵配置
├─ tests/                    # CPU 自动化测试
├─ docs/                     # 实验协议、数据格式、验证与版本说明
├─ segment/                  # 原分割代码、网络结构及本地 VOC 数据
├─ Classifier_ACC0.93/       # 原分类代码及历史特征文件
├─ artifacts/                # 清单、划分、特征和实验计划（Git 忽略）
├─ runs/                     # 模型、日志、预测和指标（Git 忽略）
├─ paper_output/             # 论文表格与图件（Git 忽略）
├─ environment.yml           # Conda 环境基础定义
├─ requirements.txt          # Python 依赖（PyTorch 需按主机单独安装）
└─ README.md
```

### `riverlab` 模块

| 文件 | 作用 |
| --- | --- |
| `cli.py` / `__main__.py` | 命令行解析及子命令入口 |
| `data.py` | 数据扫描、事件关联、分组划分及泄漏检查 |
| `masks.py` | 不同 mask 编码的统一解码 |
| `models.py` | DeepLabV3+、U-Net、SegNet、LR-ASPP、SegFormer 和分类模型定义 |
| `segmentation.py` | 分割训练、验证、测试、滑窗推理和速度测试 |
| `augment.py` | 仅训练集使用的天气、设备和翻转增广 |
| `reconstruct.py` | 将白天 `5×2` 裁剪块重建为原图分析 ROI |
| `features.py` | 面积、边界、质量和时序特征提取，历史 CSV 审计 |
| `classification.py` | 阈值、逻辑回归、随机森林、MLP 和双分支 CNN |
| `pipeline.py` | 图像对的分割—特征—分类完整推理 |
| `replay.py` | 连续监测事件回放、漏报、误报和延迟统计 |
| `metrics.py` | 分割、分类、边界和 bootstrap 指标 |
| `reporting.py` | 混淆矩阵、PR 曲线和结果汇总 |
| `suite.py` | 将实验矩阵展开为独立配置与命令 |
| `io.py` | 路径、哈希、运行记录和安全权重加载 |

### 主要配置

| 文件 | 作用 |
| --- | --- |
| `segmentation.json` | 夜间整图分割基线 |
| `segmentation_day.json` | 白天重建 ROI 分割基线 |
| `experiments.json` | 分割/分类主对比矩阵 |
| `optional_experiments.json` | 数据增广和损失函数消融矩阵 |
| `classification.json` | 堵江分类基线 |
| `day_dataset.json` / `night_dataset.json` | 日夜数据扫描配置 |
| `quality.json` | 图像和几何质量门控参数 |

## 环境安装

CPU 本地环境用于数据检查、重建和测试；正式分割训练建议使用 CUDA GPU。PyTorch 和 torchvision 必须按远程服务器的 CUDA 驱动单独安装，不要直接复制本机 CPU 版本。

```bash
conda env create -f environment.yml
conda activate river-research

# 按服务器 CUDA 版本安装 torch/torchvision 后：
python -m pip install -r requirements.txt
python -m riverlab doctor
python -m unittest discover -s tests -v
```

`doctor` 输出的 `cuda_available` 在 GPU 服务器上应为 `true`。

## 数据准备

### 夜间分割

夜间主实验直接读取：

```text
segment/VOCdevkit/Full_size/JPEGImages/
segment/VOCdevkit/Full_size/SegmentationClass/
artifacts/night_split/manifest.csv
```

检查命令：

```bash
python -m riverlab check-manifest \
  --manifest artifacts/night_split/manifest.csv \
  --mask-samples 999999
```

### 白天分割

白天 VOC2007 是 `5×2` 裁剪块，先按 `yx` 编号重建为 `2560×1024` ROI：

```bash
python -m riverlab reconstruct \
  --manifest artifacts/day_split/manifest.csv \
  --output artifacts/day_roi \
  --columns 5 \
  --rows 2 \
  --layout yx

python -m riverlab check-manifest \
  --manifest artifacts/day_roi/manifest.csv \
  --mask-samples 999999
```

重建目录约占十几 GB，可以在本地 CPU 完成后传到 GPU 服务器，也可以上传约 4.3 GB 的 VOC2007 后在服务器重建。

## 模型对比与消融实验

夜间模型对比默认包含 DeepLabV3+ MobileNet、U-Net、LR-ASPP MobileNet，每个模型使用 3 个随机种子：

```bash
python -m riverlab plan \
  --config configs/experiments.json \
  --output artifacts/experiment_plan

python -m riverlab train-seg \
  --config artifacts/experiment_plan/seg_main_000.json \
  --output runs/seg_main_000 \
  --device cuda
```

`seg_main_000` 至 `seg_main_008` 是 9 次分割主对比。`commands.txt` 后续还包含分类实验，在分类特征尚未生成时不要整文件执行。

数据增广消融包含无增广、仅天气、仅设备、天气与设备同时使用；损失消融包含 CE、CE+Dice、CE+Dice+边界损失：

```bash
python -m riverlab plan \
  --config configs/optional_experiments.json \
  --output artifacts/optional_plan
```

每次训练完成后必须使用 `best.pt` 评估相同的固定测试集：

```bash
python -m riverlab eval-seg \
  --checkpoint runs/seg_main_000/best.pt \
  --manifest artifacts/night_split/manifest.csv \
  --split test \
  --output runs/seg_main_000_test \
  --device cuda
```

正式结果按三个种子报告均值和标准差，并分别保存测试集 mIoU、Dice、像素准确率、边界 F1、逐图预测和运行配置。训练集可做随机增广，验证集和测试集不做增广。

## 输出文件

| 输出 | 内容 |
| --- | --- |
| `run.json` | 配置、代码版本、输入哈希和运行状态 |
| `best.pt` | 验证集选择的最佳分割模型 |
| `model.joblib` | 分类模型、预处理器和冻结阈值 |
| `history.csv` | 每轮训练/验证记录 |
| `metrics.json` | 测试集整体指标 |
| `per_image.csv` / `predictions.csv` | 逐样本结果 |
| `cohorts.json` | 日夜、天气、月份等分层结果 |

完整实验协议见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)，字段说明见 [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md)。

## 上传到远程 Git 仓库

当前仓库已配置 `origin`。推送前先检查工作区和提交内容：

```bash
git status
git diff --check
git log --oneline -5
```

确认提交中不含数据、权重、凭据和运行产物后推送：

```bash
git push -u origin master
```

首次使用 HTTPS 推送时，GitHub 通常要求浏览器登录或 Personal Access Token；不要把 Token 写入仓库、脚本或 README。也可以在个人电脑完成 SSH Key 配置后改用 SSH 远程地址。

GPU 服务器获取代码：

```bash
git clone https://github.com/lbai-2006/river_warning.git
cd river_warning
```

后续同步代码：

```bash
git pull --ff-only
```

### 数据和权重的传输

以下内容已被 `.gitignore` 排除，Git 推送不会携带它们：

- `segment/VOCdevkit/`
- `artifacts/`
- `runs/`
- `paper_output/`
- `*.pt`、`*.pth`、`*.joblib`
- 图像、视频、压缩包和 CSV 数据

这些文件应通过校内存储、`scp`、`sftp` 或 `rsync` 单独传输，并保持仓库内相对路径。例如夜间模型对比至少需要：

```text
segment/VOCdevkit/Full_size/
artifacts/night_split/
```

白天模型对比需要以下二选一：

```text
segment/VOCdevkit/VOC2007/ + artifacts/day_split/
```

或直接传输已经重建的：

```text
artifacts/day_roi/
```

传输完成后先运行 `check-manifest`，再启动训练。不要通过取消 `.gitignore` 的方式把私有数据和十几 GB 的运行结果强行提交到普通 Git 仓库；若将来确需版本化大权重，应单独评估 Git LFS 或对象存储。

## 文档索引

- [实验运行顺序与命令](docs/EXPERIMENTS.md)
- [数据清单与结果字段](docs/DATA_SCHEMA.md)
- [CPU 验证和数据检查](docs/CPU_VERIFICATION.md)
- [旧分类特征核对与修复](docs/LEGACY_FEATURE_RECONCILIATION.md)
- [Git 版本与迁移说明](docs/REPOSITORY.md)

## 当前状态

- 实验代码和 CPU 自动化测试已完成并通过验证。
- 白天、夜间分割数据清单及分组划分已生成。
- 夜间模型对比和增广/损失消融配置已生成，等待 GPU 运行。
- 白天 ROI 重建流程和配置已就绪，生成 `artifacts/day_roi` 后可运行白天模型对比。
- 论文性能数字必须来自固定测试集的实际运行结果，历史目录名或旧准确率不作为正式结论。
