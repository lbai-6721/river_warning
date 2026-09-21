# 河道图像监测实验工程

本仓库将已有日夜河体分割、几何变化与堵塞分类代码整理为可复现的研究工程。原始数据、标签及权重保留在本地，不提交 Git。

## 目录

| 目录 | 用途 |
| --- | --- |
| `riverlab/` | 新的统一命令行、数据检查、训练与评估模块 |
| `configs/` | 可迁移的实验配置 |
| `tests/` | CPU 自动化验证 |
| `docs/` | 实验说明、数据格式和版本记录 |
| `segment/` | 既有分割实现及本地数据；保留原路径以免破坏历史记录 |
| `Classifier_ACC0.93/` | 既有分类实现及 CSV；目录名不是有效性能证明 |
| `artifacts/` | 自动生成的数据索引、划分与特征，Git 忽略 |
| `runs/` | 独立运行目录、配置、预测与模型，Git 忽略 |
| `paper_output/` | 从实际结果生成的表格和图，Git 忽略 |

先使用新流程检查数据和修复评价协议，再运行正式训练。不要直接把旧脚本输出当成独立测试结果。

## 运行入口

```powershell
Set-Location 'D:\scientific_research\river-project\codes\river_warning'
$riverPython = 'D:\anaconda\envs\river-segment\python.exe'
& $riverPython -m riverlab doctor
& $riverPython -m unittest discover -s tests -v
& $riverPython -m riverlab --help
```

- [实验实现、GPU 运行顺序与命令](docs/EXPERIMENTS.md)
- [数据清单、事件标注与结果格式](docs/DATA_SCHEMA.md)
- [本次 CPU 验证与数据检查](docs/CPU_VERIFICATION.md)
- [Git 版本与迁移说明](docs/REPOSITORY.md)

**当前数据待办：** 旧三张分类 CSV 的243个图像对中228个标识核对不通过，不能直接拼接后训练。使用 `audit-legacy-features` 查看错配行，核对标签来源后从原图重算特征。原始数据保持原样。

正式模型训练与论文性能结果尚未完成；当前交付为通过 CPU 验证的实验工程。
