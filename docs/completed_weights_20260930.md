# 已完成训练的最佳权重清单（2026-09-30）

共 13 份训练运行已结束且 `best.pt` 可核验。9 月 30 日新下载 10 份；白天整图早停的 3 份本地已存在，经 SHA-256 核对一致，未重复下载。最佳轮次是验证损失最低的保存轮次，并非最后训练轮次。

| 实验 | 模型 | 完成轮数 | 最佳轮次 | 最佳验证损失 | SHA-256 | 服务器权重 | 本地状态 |
|---|---|---:|---:|---:|---|---|---|
| 白天整图早停 | DeepLabV3+ MobileNetV2 | 30 | 10 | 0.05981 | `993864c023e2b6f308f42f25d89c0a40f07f79a39c05315ecfcbac484b6d8061` | `runs/day_full_scratch_early_stop_000/best.pt` | 此前已有，哈希一致 |
| 白天整图早停 | UNet | 36 | 16 | 0.05076 | `4a9ff00289a7f20b5b6a8a21eda350d5722da820cb0d50789950335e5babc851` | `runs/day_full_scratch_early_stop_001/best.pt` | 此前已有，哈希一致 |
| 白天整图早停 | LRASPP MobileNetV3Large | 48 | 28 | 0.05572 | `a72509fb8a9435eadd4f7da67f2cf0df564ab411f11118de2617ea112aaff923` | `runs/day_full_scratch_early_stop_002/best.pt` | 此前已有，哈希一致 |
| 白天分块100轮 | LRASPP MobileNetV3Large | 100 | 3 | 0.03813 | `90af80dd99631e36bdbe344baa552ebf9eebb0e18ad76115573c40c71f179726` | `runs/day_patch_100_parallel_20260929_lraspp/best.pt` | 本次下载，哈希一致 |
| 白天整图100轮 | DeepLabV3+ MobileNetV2 | 100 | 80 | 0.06103 | `e906daf6da6f76dc7561b052c4b607a13af5558bb1067355864192ed56e191ea` | `runs/day_roi_100_parallel_20260929_deeplab/best.pt` | 本次下载，哈希一致 |
| 白天整图100轮 | LRASPP MobileNetV3Large | 100 | 28 | 0.05224 | `04e81616b7bf354727c6263526e0ea3abbedec2008430e74c7012faf7ff2236b` | `runs/day_roi_100_parallel_20260929_lraspp/best.pt` | 本次下载，哈希一致 |
| 白天整图100轮 | UNet | 100 | 46 | 0.05162 | `57a2c4c294839ad0213e9054e7fdb30937c9bf0fe7fef497e1abed58d6ce581e` | `runs/day_roi_100_parallel_20260929_unet/best.pt` | 本次下载，哈希一致 |
| 夜间重建整图100轮 | DeepLabV3+ MobileNetV2 | 100 | 42 | 0.04427 | `75e085365eb8ca785731780fc55bb6427b37f7af010a2ae25d4594ea2eac7d93` | `runs/night_reconstructed_scratch_no_early_stop_000/best.pt` | 本次下载，哈希一致 |
| 夜间重建整图100轮 | UNet | 100 | 38 | 0.05423 | `4f49ab5c6aa5434629872d69af9de3955091cbe8d6d7e24599eac22b0914d7f6` | `runs/night_reconstructed_scratch_no_early_stop_001/best.pt` | 本次下载，哈希一致 |
| 夜间重建整图100轮 | LRASPP MobileNetV3Large | 100 | 46 | 0.04906 | `878184ed392071bdfd582d67c447cf70b0221731dd16ebdf83f0828ad4f95c7d` | `runs/night_reconstructed_scratch_no_early_stop_002/best.pt` | 本次下载，哈希一致 |
| 夜间旧拆分早停 | DeepLabV3+ MobileNetV2 | 34 | 14 | 0.03648 | `5cb06c32565af51eae39034b5ebbdf26dd1014ecf46292d12ef971e3eb67aea3` | `runs/night_seg_scratch_000/best.pt` | 本次下载，哈希一致 |
| 夜间旧拆分早停 | UNet | 40 | 20 | 0.02520 | `0f079d7521d46bdcdd53a3ce97632da17f7c4d6074eb6e16faa68bd2fdc99ed9` | `runs/night_seg_scratch_001/best.pt` | 本次下载，哈希一致 |
| 夜间旧拆分早停 | LRASPP MobileNetV3Large | 31 | 11 | 0.04362 | `3fcdc66c7c3e111d1d5f566257ce1301cafd9ed9d0a13b7fd8d477a6e0be126e` | `runs/night_seg_scratch_002/best.pt` | 本次下载，哈希一致 |

本地权重位于 `D:/scientific_research/river-project/远程权重/AutoDL_已完成训练_20260930`，该目录的 `权重清单.md` 提供可点击的本地文件索引。服务器的 `runs/` 目录不进入 Git。

白天分块 UNet（36/100 轮）和 DeepLab（0/100 轮）尚未完成，故未列入。完成训练的分块 LRASPP 尚无独立测试结果。

详细评估见 [分割实验评估汇总](day_100_evaluation_20260929.md)。
