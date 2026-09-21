# 数据清单和实验产物

所有 CSV 采用 UTF-8（允许 BOM），JSON 用 UTF-8。路径支持绝对路径或相对仓库根目录；远程迁移应优先用相对路径。带空格的命令行路径需加引号。示例均是格式示范，不是已确认事件或性能。

## 1 分割图像清单

| 字段 | 含义 |
| --- | --- |
| sample_id | 唯一样本 ID，裁剪图有各自 ID |
| parent_id | 带摄像头前缀的原图 ID；同源裁剪必须相同 |
| camera_id / mode | 相机标识 / day 或 night；不能把不同时区混用 |
| timestamp | 本地统一时区下 `YYYY-MM-DDTHH:MM:SS`，不用混合格式 |
| image_path / mask_path | 原图与人工语义 mask 文件 |
| width / height | 图像尺寸；快速 inventory 中留空，`--check-sizes` 可生成 |
| event_id | 已核验事件 ID；未知留空，不按日期自动造事件 |
| weather | 经过核验的 clear/fog/rain/occlusion 等标签；未核验用 unknown |
| label / label_source | 逐时刻状态标签及其依据；仅分割数据可为空 |
| image_sha256 | 可选文件哈希；精确重复检查，非感知去重 |
| group_id / split | 划分命令生成；split 为 train/val/test |

内部统一 mask 是二维类别索引：0=背景、1=河体、255=不计分区域。输入三通道标注必须显式声明解码映射，不能直接作为类别索引。前景若使用255，必须依据真实标注定义配置，不能猜测。

本项目白天抽检发现 RGB 红色 `(128,0,0)` 与重复索引 `(1,1,1)` 两种前景编码。`configs/day_mask_mapping.json` 把它们统一为河体1，黑色为背景；红色定义与既有 VOC 类别1色表一致。该路径由 day_dataset 配置写入清单的 `mask_mapping` 字段，读取时转换，不修改原 PNG；未知颜色直接报错。夜间 P 模式0/1直接读取。正式训练前应进一步复核标注来源与语义，不能由颜色映射推断所有文件都是人工真值。

库存检查默认抽样100张 mask，正式训练前可用 `check-manifest --mask-samples 999999` 全量检查（同时逐图训练也会检查类别）。大小不匹配直接报错。

## 2 图像对清单

关键字段为 `sample_id,frame_a,frame_b,camera_id,mode,timestamp_a,timestamp,label,label_source`，另外保留 `image_a,image_b,event_id,weather`。a 是过去，b 是当前，要求 timestamp_a < timestamp。frame_a/frame_b 必须与整图清单 parent_id 完全一致；后续 joint-split 添加 group_id/split。

`pair-inventory` 只扫描指定目录下一级样本子文件夹，每夹至少两张带时间戳的图像。目录标签自动注明未核验。跨多日的图像对会把这些日期合并为一组；如果它们形成一个大连通组，不应拆开绕过泄漏检查，而应复核采样方式。

正常/异常标签描述的是论文明确约定的状态，不能从水面变化的算法输出反过来制造监督标签。训练配置 `allow_unverified_labels` 默认 false；探索旧数据时开启必须在报告里注明。

## 3 事件和监测区间

事件文件至少需要下列列名，示例：

```csv
event_id,camera_id,start,end,reviewed,evidence
example_event,day,2022-06-01T12:00:00,2022-06-01T14:00:00,1,reviewed_record_reference
```

`reviewed=1` 表示实际完成核验，不是让程序接受数据的占位开关。start 是依据证据定义的事件起始时刻，end 是结束时刻；保存 evidence 可追溯到监测记录或人工审核资料。同一真实事件在不同相机下应使用同一个 event_id 以避免跨集合。

监测文件：

```csv
camera_id,start,end,state,reviewed
day,2022-06-01T00:00:00,2022-06-01T12:00:00,normal,1
day,2022-06-01T12:00:00,2022-06-01T15:00:00,event,1
day,2022-06-01T15:00:00,2022-06-02T00:00:00,unknown,1
```

区间采用左闭右开，同摄像头不得重叠。事件起止必须被一个 event 监测区间完整包含。实际评价只传测试期间的事件/监测资料；为避免测试外事件被当成漏报，应先按事先约定的测试时间范围筛选事件。跨切分边界的事件必须在分组时保持完整。

跨相机同一事件目前回放按“相机—事件”记录计数，**不是跨相机去重后的灾害事件总数**；如果两台摄像头共同观测同一次真实事件，论文中应分别报告相机级检测和人工核对后的唯一事件统计。

## 4 特征与来源旁文件

CSV 同名 `.meta.json` 记录 source、分割 checkpoint_sha256/provenance、bins、sigma、质量参数和历史窗口配置。不要只复制 CSV 而丢掉旁文件。

来源包括：

- `predicted`：来自冻结分割权重，正式部署流程。
- `ground_truth`：人工 mask 条件对照。
- `legacy_unverified`：旧 CSV 的原量纲与未追溯来源，只供探索。
- `inventory`：数据清单，不是已计算特征。

面积字段 `area_00…`，上下边界位移 `upper_00/lower_00…`，绝对位移 `upper_abs_00/lower_abs_00…`，宽度变化 `width_00…`。缺失值为 NaN，填补器仅在训练集拟合。特征中保留 quality_valid 和 edge_coverage，后者表示两个端点各自河体可见列覆盖率的较小值，不是模型置信度。

`temporal-features` 给字段追加 `_lag1` 等后缀，并保存真实时间差。多时刻差分的上下边界是条带均值之差；它不能等价替代两帧流程中“逐列绝对位移再平均”的特征。

## 5 运行产物和复现

| 产物 | 用途 |
| --- | --- |
| run.json | 配置、代码提交、是否有未提交修改、环境、输入文件哈希、started/complete |
| best.pt / model.joblib | 最佳分割模型 / 分类模型与训练预处理、阈值 |
| history.csv | 每轮验证记录；分割另有训练损失 |
| validation_metrics.json | 分类验证结果，不是测试指标 |
| metrics.json / cohorts.json | 测试整体、按日夜/天气/月分层指标 |
| predictions.csv / per_image.csv | 逐样本预测、分割逐图结果 |
| predictions.meta.json | 分割输出权重和数据划分来源 |
| events.csv / alerts.csv / timeline.png | 回放逐事件结果、告警和时间轴 |

status=started 而没有 complete 表示没有正常结束，可能是异常或进程被中断，不能当成功实验。输入文件哈希默认针对清单、配置所指权重，不是全部原始像素的内容证明；若要完整数据版本校验，请启用库存图像哈希并保存原始数据的备份校验记录。

生成数据与权重均被 Git 忽略，不能把一次 Git 提交当作已经备份全部数据。原始数据和运行产物需按团队方案另行本地备份。
