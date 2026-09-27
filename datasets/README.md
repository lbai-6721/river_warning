# 数据集目录

本目录集中保存项目后续可能使用的全部本地数据。除本说明外，目录内容均被 Git 忽略；向 GPU 服务器迁移时直接传输整个 `datasets/`，并将它放在仓库根目录。

```text
datasets/
├─ segmentation/
│  ├─ day/
│  │  ├─ source_patches/       # 原 VOC2007：32,270 个白天裁剪块
│  │  ├─ source_split/         # 裁剪块固定划分、5 折清单和审计
│  │  └─ reconstructed_roi/    # 3,227 张重建 ROI、mask 和固定划分
│  └─ night/
│     ├─ source_patches/       # 原 VOC2023：22,700 个夜间裁剪块
│     ├─ full_size/            # 722 张夜间整图及 mask
│     └─ full_size_split/      # 夜间固定划分、5 折清单和审计
└─ classification/
   ├─ day/                     # 白天正常/堵江图像对
   ├─ night/                   # 夜间正常/堵江图像对
   └─ legacy_features/         # area.csv、up.csv、down.csv
```

当前目录合计约 18.9 GB。训练配置使用仓库根目录下的相对路径，因此服务器端不要改变上述层级。

完整性检查：

```bash
python -m riverlab check-manifest \
  --manifest datasets/segmentation/day/reconstructed_roi/manifest.csv \
  --mask-samples 999999

python -m riverlab check-manifest \
  --manifest datasets/segmentation/night/full_size_split/manifest.csv \
  --mask-samples 999999
```
