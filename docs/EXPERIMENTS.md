# EI 论文实验实现与运行说明

实现日期：2026-09-22。工程根目录：`D:\scientific_research\river-project\codes\river_warning`。

本次完成实验代码、CPU 验证和数据核查，没有在真实数据上训练，也没有产生可写进论文的模型性能结果。后续 GPU 实验使用这些入口；事件真值和标签依据仍需人工核对。

## 1 已实现的实验与边界

| 原路线编号 | 实验 | 实现入口与产物 |
| --- | --- | --- |
| E0 | 数据、原图、日期、共享帧和事件分组 | `inventory`、`pair-inventory`、`joint-split`、`split`、`check-manifest`；清单和交集审计 |
| E1 | 可信训练、验证与测试 | `train-seg`、`train-cls`、`eval-seg`、`eval-cls`；固定阈值、逐样本预测、权重来源、Git SHA |
| E2 | 分割主对比 | DeepLabv3+ MobileNet/Xception、标准 U-Net、SegNet、LR-ASPP MobileNet；可选 SegFormer-B0 |
| E3 | 分类主对比 | 面积变化阈值、逻辑回归、随机森林、双分支 CNN、MLP |
| E4 | 特征与模块消融 | 面积/边界/二者/绝对位移；`--bins`、`--sigma`、历史 `--lags`；质量门控 |
| E5 | 分割误差传递 | 人工 mask 与预测 mask 特征分别训练、测试分类器；同一划分配对比较 |
| E6 | 连续时序回放 | `replay`：事件召回、有效正常监测时长、误报/天、首次告警延迟、告警段、无有效观测事件、时间轴图 |
| E7 | 日夜、天气、月份分层 | 分割和分类的 `cohorts.json`，含样本数/独立组数；分类另含覆盖率 |
| E8 | 效率 | `benchmark`：分割与几何子流程；`infer-pairs`：完整图像对流程中位数/P95和 CUDA 峰值显存 |
| E9 | 裁剪及增强 | 整图缩放、训练随机裁剪、非重叠/重叠滑窗；基础/天气/设备增强的配置矩阵 |
| E10 | 损失对照 | CE、Focal、Dice、简单有限差分边界项；组合配置可独立控制 |

本轮轻量对照使用可在当前环境验证的 LR-ASPP，**未实现 BiSeNet V2**。SegFormer-B0 已有可选适配代码，但本机缺少 transformers，未运行验证。正式选用前应在 GPU 环境补测。边界损失是本工程的简单实现，不冒用其他论文特定算法名称。

新 U-Net 和双分支 CNN 都是结构明确的基线；不声称复现旧草稿或旧权重的全部细节。所有模型默认随机初始化、禁止自动下载权重；如设置 `initial_weights`，严格检查参数匹配并记录文件哈希。比较预训练模型时，另行统一预处理和初始化协议，不能仅加载权重就称公平比较。

## 2 本机使用方式

在 PowerShell 中：

```powershell
Set-Location 'D:\scientific_research\river-project\codes\river_warning'
$riverPython = 'D:\anaconda\envs\river-segment\python.exe'
& $riverPython -m riverlab doctor
& $riverPython -m unittest discover -s tests -v
& $riverPython -m riverlab --help
```

实际找到的环境名是 `river-segment`，不是 `river`。测试只在临时生成的小图和数值样本上执行少量优化步骤、保存与加载权重；不训练项目数据，不下载模型。临时测试数据自动清理。

以下相对路径全部以仓库根目录为基准。运行目录禁止覆盖，重跑时换目录名。训练入口默认请求 CUDA，无 CUDA 会明确报错，不会悄悄在 CPU 上跑长期训练；逻辑回归、随机森林、面积阈值本身使用 CPU。

## 3 先核查数据，避免继续使用旧划分

```powershell
& $riverPython -m riverlab audit-legacy --output artifacts/legacy_split_audit.json
& $riverPython -m riverlab audit-legacy-features --output artifacts/legacy_feature_audit.json
& $riverPython -m riverlab inventory --config configs/night_dataset.json --output datasets/segmentation/night/full_size_manifest.csv
& $riverPython -m riverlab inventory --config configs/day_dataset.json --output datasets/segmentation/day/source_manifest.csv
```

旧 CSV 共243对。早期审计把旧标识中的连字符和紧凑日期格式也判错，原先报告的228对不是228对数值错误。修正解析后，134对起止小时一致，103对仅起始小时不同，另4对标识不完整、2对双端小时不一致。核对原始图像文件夹发现103对正例的同名文件夹包含边界表所列两张图，因此多数是目录/标识命名问题；面积特征数值和标签依据仍需追溯。2021年有103正/15负，2022年只有125负，跨年份分类会受标签分布影响。逐行复核见 `paper_output/data_cleaned/legacy_pair_reconciliation.csv`；`import-legacy` 发现未复核问题仍拒绝导入，原 CSV 不改写。

夜间清单已核对 722 张图及同名 mask。本次按日期先后生成的独立分割试运行划分为训练 457、验证 124、测试 141 张，对应 36/12/12 个日期组。这是**未加入事件清单的开发候选划分**，正式事件标注完成后应在首次正式训练之前重新冻结全项目划分。

```powershell
& $riverPython -m riverlab split --manifest datasets/segmentation/night/full_size_manifest.csv --output datasets/segmentation/night/full_size_split_v2 --folds 5
& $riverPython -m riverlab check-manifest --manifest datasets/segmentation/night/full_size_split_v2/manifest.csv
```

`split` 默认按时间顺序，60%/20%/20% 针对独立组，不保证图像数恰为该比例。它会合并同原图裁剪、共享帧、同日期（跨摄像头）、同已确认事件与相同图像哈希。`--hash-images` 可计算精确文件哈希；它不能识别重新压缩、改尺寸后的近重复图像，仍需人工核查。

`--folds 5` 只对开发数据做分组交叉验证，测试集固定。交叉验证允许过去和未来日期在开发折内交换，属于回顾性开发；主测试仍为较晚时段。不能反复根据测试结果挑模型或挑随机种子。跨多日事件导致日期交错时，时间划分直接报错；先核对事件边界，不为追求分数随意拆组。

库存默认快速核对文件配对、标识和时间，width/height 留空，不代表已逐像素核验。`inventory --check-sizes` 逐图核对尺寸；`check-manifest` 抽查图像尺寸和 mask 值，设置足够大的 `--mask-samples` 可全查。训练读取每个样本时也会检查尺寸和类别。这样后续反复生成索引无需读取全部图片。

## 4 分类数据与分割必须一起划分

对确认用途的图像对目录运行：

```powershell
& $riverPython -m riverlab pair-inventory --root PATH_TO_REVIEWED_NORMAL_FOLDER --camera day --mode day --label 0 --output artifacts/normal_pairs.csv
& $riverPython -m riverlab pair-inventory --root PATH_TO_REVIEWED_ABNORMAL_FOLDER --camera day --mode day --label 1 --output artifacts/abnormal_pairs.csv
& $riverPython -m riverlab merge-manifests --inputs artifacts/normal_pairs.csv artifacts/abnormal_pairs.csv --output artifacts/pairs_inventory.csv
```

目录标记只用于整理。自动生成 `label_source=directory_inventory_unverified`；核实每条标签并填写实际依据（例如 `reviewed:record_001`）。不能把目录名自动当作真实堵江事件。四帧文件夹仅生成相邻图像对，不生成任意两两组合。

合并需要使用的日夜分割清单，再将所有图像对、分割图和事件共同分组：

```powershell
& $riverPython -m riverlab merge-manifests --inputs datasets/segmentation/day/source_manifest.csv datasets/segmentation/night/full_size_manifest.csv --output artifacts/all_frames.csv
& $riverPython -m riverlab joint-split --frames artifacts/all_frames.csv --pairs artifacts/pairs_inventory.csv --events artifacts/reviewed_events.csv --output artifacts/joint
```

尚无确认事件时去掉 `--events`；此时仅能开展日期分组的状态识别实验。输出 `frames.csv`、`pairs.csv` 和审计。检查每个训练/验证子集是否都有正负类别；程序不通过移动测试样本补齐类别。若独立正例组不足，应补数据或降低结论范围。

同一分割模型生成下游特征时，分类验证组不能进入分割训练，分类测试组不能进入分割训练或验证。代码不仅检查 group_id，还检查原图标识、日期和事件，防止在另一个清单中改名后绕过检查。

## 5 分割训练和整图评估

以下训练命令留到 GPU 可用后执行，先按冻结清单修改配置中的 `manifest`：

```bash
python -m riverlab train-seg --config configs/segmentation.json --output runs/night_deeplab_s42 --device cuda
python -m riverlab eval-seg --checkpoint runs/night_deeplab_s42/best.pt --manifest datasets/segmentation/night/full_size_split/manifest.csv --output runs/night_deeplab_s42_test --device cuda
```

默认夜间整图缩放训练，在原图尺寸恢复概率图后计算 IoU、Dice、边界 F1，255 作为无效标签忽略。边界容差以原图像素计，默认 2 像素；比较不同图像分辨率时需预先确定统一规则。输出分组 bootstrap 置信区间，不把每个裁剪块当成独立重复。

白天已有图是 5 列×2 行裁剪，编号为行列 `00…04,10…14`，重建默认 `--layout yx`；不要把它读成列行。输入先按原图分组。白天源 mask 混有 RGB 编码，day_dataset 配置已显式指定解码映射；未知颜色会报错，原图不改写。可在 GPU 主机有足够存储后执行：

```bash
python -m riverlab split --manifest datasets/segmentation/day/source_manifest.csv --output datasets/segmentation/day/source_split
python -m riverlab reconstruct --manifest datasets/segmentation/day/source_split/manifest.csv --output datasets/segmentation/day/reconstructed_roi --columns 5 --rows 2
python -m riverlab train-seg --config configs/segmentation_day.json --output runs/day_deeplab_s42 --device cuda
python -m riverlab eval-seg --checkpoint runs/day_deeplab_s42/best.pt --manifest datasets/segmentation/day/reconstructed_roi/manifest.csv --output runs/day_deeplab_s42_test --device cuda
```

上述白天独立划分用于分割对比；端到端论文实验应换成 `joint-split` 的同一划分。`reconstruct` 必须输入同一来源的完整裁剪网格，不能直接混入夜间整图；混合清单先按来源筛出白天 patch 行。它只能重建既有裁剪覆盖的 ROI，不能恢复原来被丢掉的画面。缺块、重复块或同原图跨集合会报错。

`segmentation_day.json` 配置 512 输入、512 窗口、384 步长；非重叠对照改步长为 512；整图缩放对照设置 `train_crop=false,tile_size=0`。几何特征必须从整图/重建 ROI mask 提取，不能把独立 patch 边界当成河岸。

训练只在训练集更新参数，按验证损失早停并保存最佳权重；每轮恢复 train 模式。测试入口不调阈值、不反传。`--limit` 仅用于推理烟雾检查，指标标记为 `limited_smoke_only`，不能作为论文正式结果。

## 6 几何特征、分类与误差传递

对同一个已冻结分割模型，依次导出 train、val、test 预测，合并三张 `predictions.csv`。三者必须来自同一权重、同一输入清单与推理配置：

```bash
python -m riverlab eval-seg --checkpoint runs/seg/best.pt --manifest artifacts/full_frames.csv --split train --output runs/seg_train --device cuda
python -m riverlab eval-seg --checkpoint runs/seg/best.pt --manifest artifacts/full_frames.csv --split val --output runs/seg_val --device cuda
python -m riverlab eval-seg --checkpoint runs/seg/best.pt --manifest artifacts/full_frames.csv --split test --output runs/seg_test --device cuda
python -m riverlab merge-manifests --inputs runs/seg_train/predictions.csv runs/seg_val/predictions.csv runs/seg_test/predictions.csv --output artifacts/predictions.csv
python -m riverlab pair-features --pairs artifacts/joint/pairs.csv --frames artifacts/predictions.csv --output artifacts/features/manifest.csv --bins 20 --sigma 0 --quality-config configs/quality.json
python -m riverlab train-cls --config configs/classification.json --output runs/cls_logistic_s42 --device cuda
python -m riverlab eval-cls --checkpoint runs/cls_logistic_s42/model.joblib --features artifacts/features/manifest.csv --output runs/cls_logistic_s42_test
python -m riverlab plot-classifier --predictions runs/cls_logistic_s42_test/predictions.csv --output paper_output/cls_logistic_s42
```

`runs/seg` 等名称是示范路径，需换成实际运行目录。`full_frames.csv` 指按联合划分整理好的整图/重建 ROI 清单；混用单块 mask 会被拒绝。正式分类默认要求标签来源已核验；旧数据探索可单独设 `allow_unverified_labels=true`，但不得据此宣称验证了真实事件。

人工 mask 对照使用 `pair-features --ground-truth --frames FULL_FRAME_MANIFEST`，保存另一套特征、配置和分类运行目录。它是人工掩码条件参考，不是部署结果。代码拒绝训练用人工 mask、测试突然换预测 mask 的混用；若要研究这种分布变化，应另设明确协议。

几何量定义：

- 20 个固定竖条全部保留；面积为条带内有效像素的河体占比，不是平方米。
- upper/lower 是图像坐标中的上/下包络，不等同于水文上下游。坐标、位移除以图像高度归一化；不同相机视角或 ROI 不因此自动可比。
- 同时保存带符号位移与先逐列取绝对值、再平均的位移。没有河体的列保留缺失，不伪装成零位移。
- `--sigma` 是横向平滑的像素标准差，0 表示不平滑；分区数量和平滑参数只能根据验证集选择。
- 训练集拟合缺失值填补和标准化；验证/测试只应用。指标用完整预测计算，不平均每批准确率。
- 阈值默认在验证集最大化 F1，平局取较高阈值；也可预先指定固定阈值。平均精度字段为 `average_precision`，不是梯形积分 PR-AUC。

分类 `feature_set` 可为 `area`、`boundary`、`both`、`geometry`。`geometry` 加入绝对边界位移和宽度变化；双分支 CNN 仅适合有面积和等长上下边界的 `both` 配置，面积/边界单独消融用逻辑回归或 MLP。

质量门控先根据开发数据填写 `quality.json`，再统一提取全部划分；默认阈值全部放行，并不代表质量控制已有效。分类配置设 `quality_gate=true`，可另设 `edge_coverage_min`。结果同时报告接受样本性能、覆盖率、被拒绝正例数，以及拒绝后不报警的总体性能，不能只报告筛选后变高的准确率。

## 7 多时刻、事件回放与效率

有连续整图和逐时刻标签时：

```bash
python -m riverlab frame-features --manifest artifacts/predictions.csv --output artifacts/frame_features.csv --bins 20
python -m riverlab temporal-features --frames artifacts/frame_features.csv --output artifacts/temporal_features.csv --lags 1 2 3 --max-gap-hours 3
```

按摄像头、独立组和 split 分别排序，严格使用过去帧。缺历史、重复时刻或超出最大间隔的窗口被丢弃；不跨组借帧，也不使用未来插值。特征是当前与各历史帧之差，非 LSTM。间隔随样本变化时应限制采样或另定义速率特征。默认不跨日期组生成窗口，因此回放时要显式统计启动/窗口缺失造成的观测覆盖限制。连续时序模型使用对应 temporal 特征训练与 `eval-cls`；下面 `infer-pairs` 专用于两帧特征模型。

```bash
python -m riverlab infer-pairs --seg-checkpoint runs/seg/best.pt --cls-checkpoint runs/cls/model.joblib --pairs artifacts/joint/pairs.csv --frames artifacts/full_frames.csv --output runs/pipeline_test --device cuda
python -m riverlab replay --predictions runs/pipeline_test/predictions.csv --events artifacts/reviewed_events.csv --monitoring artifacts/reviewed_monitoring.csv --output runs/replay_test --consecutive 2 --max-gap-hours 2
```

`infer-pairs` 从实际图像运行分割、归一化几何、质量门控和冻结分类器，并核对分割权重哈希。它计时两幅图解码、两次分割、特征和分类；预热后汇总中位数/P95，注明无跨图像对缓存。不包括加载权重、最终写 CSV 和事件回放。不同配置应在同一硬件、相同测试图像上比较。

`replay` 要求每摄像头每时刻仅一个分数和已冻结阈值；任意抽选的重复图像对不是连续流。提供完整正常/事件/未知监测区间及事件起止证据。默认只匹配事件发生后告警；只有有依据地预先定义预警窗口，才设置 `--warning-hours`。延迟可为负值，表示在该预设窗口内提前报警。

事件召回包含没有有效图像的事件；延迟中位数只统计检测到的事件，必须和漏报率一起报告。误报/天分母为满足最大采样间隔的相邻有效观测之间、与正常区间重叠的小时数，不能把未观察的空档计为正常监测。未知区间告警单独列出。

子流程效率入口：

```bash
python -m riverlab benchmark --checkpoint runs/seg/best.pt --image PATH_TO_IMAGE --output runs/seg_speed --device cuda --warmup 5 --repeats 100
```

该入口不包括分类和文件读取，不能当完整系统延迟。CPU 没有 CUDA 显存指标，以 null 表示；本轮未实现跨平台进程峰值 RAM 采样。GPU 上输出 CUDA 峰值分配显存，并应附 GPU 型号、驱动、输入尺寸和模型配置。

## 8 GPU 实验安排与结果导出

```bash
python -m riverlab plan --config configs/experiments.json --output artifacts/experiment_plan_v2
python -m riverlab plan --config configs/optional_experiments.json --output artifacts/optional_plan
```

主矩阵生成 28 个训练配置/命令：3 分割基线×3种子，3分类基线×3种子，3特征设置×3种子和1个面积阈值。默认分类矩阵需要先完成已核验预测特征；其中逻辑回归 both 与主对比重复，是便于单独执行的对照行，汇总时不能当作额外独立重复。可选增强/损失矩阵21项；SegFormer 配置单独生成。`plan` **只生成清单，不运行训练**。

按顺序执行：一个分割配置试跑并核对产物 → 分割主对比 → 固定分割模型生成特征 → 分类主对比 → 主方法的一项改进与消融 → 事件回放与效率。先用验证结果排除无效配置，再开展关键模型三种子重复；28条命令不是必须一次全部跑完。

日间对照复制主矩阵，`base_config` 改为 `configs/segmentation_day.json` 并使用独立名称。划分在每个随机种子间保持固定；训练种子控制初始化、采样和增强。为避免碰撞，生成命令的运行名也需随数据版本修改。

每个正式训练配置、选择规则与调参预算需留档；同一默认学习率不保证各网络都调优充分。GroupKFold 已输出折清单，将 `manifest/features` 换成对应折并使用不同运行目录，即可开展开发阶段交叉验证；不要对测试集做模型选择。

```bash
python -m riverlab aggregate --runs runs/model_a_test runs/model_b_test --output paper_output/results.csv
```

每次运行保留 `run.json`（配置、Git版本、环境、输入的清单哈希）、权重、验证记录、逐样本预测和指标。`aggregate` 输出真实标量结果，不虚构尚未训练的结果。种子均值/标准差应按同一配置的多次运行汇总，不能把不同模型或不同测试集合在一起。分组 bootstrap 区间与跨种子标准差是两种不确定性，不能互相替代。

目前环境文件仅提供依赖起点，GPU 主机需要按实际 CUDA/驱动安装匹配 PyTorch，再运行 `doctor` 和 CPU 测试；未选择远程主机前，本轮不连接、不上传。数据、权重和运行产物不进 Git，迁移数据时保持相对目录或修改配置，并保存校验清单。joblib 分类权重仅加载可信来源。

## 9 还需要人工或 GPU 完成的事项

1. 确认 0/1 对应状态、原始标签依据、事件边界、连续正常监测时段、历史公开与数据权属。
2. 修复旧特征标识追溯，或从复核的图像对重新计算。补缺失的原图/ROI mask 对应关系。
3. 补天气/遮挡标签及标注抽检；当前 unknown 不能写成真实天气鲁棒性验证。
4. 冻结联合划分后，在 GPU 上执行正式训练与独立测试，保留失败案例。
5. 根据真实实验决定论文贡献；整理技术交底中的方法流程、参数和技术效果。此次代码提交不是专利新颖性结论，也不是完整软著申报材料。

软件著作权后续可在稳定算法基础上增加导入、模型管理、预警记录和结果导出的可操作界面及用户手册。现阶段交付的是实验工程和 CLI，不声称已完成可交付业务软件。
