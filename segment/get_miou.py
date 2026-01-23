import os

from PIL import Image
from tqdm import tqdm

from UUUUUUUnet import Unet
from SSSegnet import SegNet

import numpy as np
from deeplab import DeeplabV3
from Utils.utils_metrics import compute_mIoU, show_results
from PSPPPPPPnet import PspNet

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''


def calculate_miou(
        miou_mode=0,
        num_classes=2,
        name_classes=["background", "riverbed"],
        VOCdevkit_path='VOCdevkit',
        day_night_mode='day',
        model_arch='segnet'
):
    """
    封装计算mIoU的完整流程

    参数说明：
    miou_mode       - 0:完整流程 1:仅预测 2:仅计算指标
    num_classes     - 分类数+背景（实际类别数+1）
    name_classes    - 类别名称列表
    VOCdevkit_path  - 数据集根目录路径
    day_night_mode  - 数据模式（'day'/'night'）
    model_arch      - 使用的模型架构（'unet'/'deeplab'/'segnet'）
    """
    # 路径配置
    image_ids = open(os.path.join(VOCdevkit_path, "Full_size/ImageSets/Segmentation/test.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "Full_size/SegmentationClass/")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    # 模式参数校验
    assert miou_mode in [0, 1, 2], "miou_mode参数错误，应为0、1或2"
    assert day_night_mode in ['day', 'night'], "day_night_mode参数错误"
    assert model_arch in ['unet', 'deeplab', 'segnet', 'pspnet'], "不支持的模型架构"

    # 阶段1：预测结果生成
    if miou_mode in [0, 1]:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        # 模型初始化
        print("Loading model...")
        model = None
        if model_arch == 'unet':
            model = Unet()
        elif model_arch == 'deeplab':
            model = DeeplabV3()
        elif model_arch == 'segnet':
            model = SegNet()
        elif model_arch == 'pspnet':
            model = PspNet()
        print(f"{model_arch.upper()}模型加载完成")

        # 批量预测
        print("Generating predictions...")
        for image_id in tqdm(image_ids, desc="Processing Images"):
            image_path = os.path.join(VOCdevkit_path, "Full_size/JPEGImages", f"{image_id}.jpg")

            try:
                image = Image.open(image_path)
                # 统一调用接口
                if model_arch == 'deeplab':
                    pred = model.get_miou_png(image, day_night_mode)
                else:
                    pred = model.get_miou_png(image, day_night_mode)

                pred_path = os.path.join(pred_dir, f"{image_id}.png")
                pred.save(pred_path)
            except Exception as e:
                print(f"处理图像 {image_id} 时出错: {str(e)}")
                continue

    # 阶段2：指标计算
    if miou_mode in [0, 2]:
        print("\nCalculating metrics...")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            png_name_list=image_ids,
            num_classes=num_classes,
            name_classes=name_classes
        )

        # 结果显示与保存
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

        # 返回计算结果
        return {
            'histogram': hist,
            'IoUs': IoUs,
            'PA_Recall': PA_Recall,
            'Precision': Precision,
            'mIoU': np.nanmean(IoUs)
        }


if __name__ == "__main__":
    # 示例用法
    results = calculate_miou(
        miou_mode=0,
        num_classes=2,
        name_classes=["background", "riverbed"],
        VOCdevkit_path='VOCdevkit',
        day_night_mode='night',
        model_arch='deeplab'
    )
    print(f"mIoU结果：{results['mIoU']:.4f}")
