# -*- coding = utf-8 -*-
# @Time : 2024
# @File : split_image_20.py
# @Description : 将图片分割为20块（5列×4行）

import os
import cv2
import numpy as np
from PIL import Image

def split_image_20_blocks(img_path, output_dir='img/temp', block_size=512):
    """
    将图片分割为20块（5列×4行）
    
    参数:
        img_path: 输入图片路径
        output_dir: 输出目录
        block_size: 每个块的大小（默认512x512）
    """
    # 读取图片
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return
    
    # 调整图片大小（如果需要）
    # img = cv2.resize(img, dsize=(2560, 1440), interpolation=cv2.INTER_NEAREST)
    img_height, img_width = img.shape[0], img.shape[1]
    
    # 创建输出目录
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # 计算每块的尺寸
    cols = 5  # 水平方向5列
    rows = 4  # 垂直方向4行
    
    block_width = img_width // cols
    block_height = img_height // rows
    
    # 分割图片为20块（5列×4行）
    for i in range(cols):  # 水平方向：0,1,2,3,4
        for j in range(rows):  # 垂直方向：0,1,2,3
            # 计算裁剪区域
            x_start = i * block_width
            x_end = (i + 1) * block_width if i < cols - 1 else img_width
            y_start = j * block_height
            y_end = (j + 1) * block_height if j < rows - 1 else img_height
            
            # 裁剪图片块
            img_temp = img[y_start:y_end, x_start:x_end, :]
            
            # 保存图片块
            if img_temp.size > 0:
                filename = f"{j}{i}.jpg"  # 文件名格式：行_列.jpg (如: 00.jpg, 10.jpg, 01.jpg, ...)
                cv2.imwrite(os.path.join(output_dir, filename), img_temp)
                print(f"保存块 [{j},{i}]: {filename}, 尺寸: {img_temp.shape}")
    
    print(f"\n图片已分割为 {cols * rows} 块，保存在: {output_dir}")


def merge_image_20_blocks(input_dir='img/temp', output_path='merged.jpg', 
                          original_size=(2560, 1440), block_size=512):
    """
    将20块图片拼接回原图
    
    参数:
        input_dir: 输入目录（包含分割后的图片块）
        output_path: 输出路径
        original_size: 原始图片尺寸 (width, height)
        block_size: 每个块的大小
    """
    cols = 5  # 水平方向5列
    rows = 4  # 垂直方向4行
    
    img_width, img_height = original_size
    block_width = img_width // cols
    block_height = img_height // rows
    
    # 创建空白画布
    merged_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # 按照切分时的顺序进行拼接
    for i in range(cols):  # 水平方向：0,1,2,3,4
        for j in range(rows):  # 垂直方向：0,1,2,3
            filename = f"{j}{i}.jpg"  # 文件名格式：行_列.jpg
            image_path = os.path.join(input_dir, filename)
            
            if not os.path.exists(image_path):
                print(f"警告: 图片块 {filename} 不存在，跳过")
                continue
            
            # 读取图片块
            block = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if block is None:
                print(f"警告: 无法读取图片块 {filename}，跳过")
                continue
            
            # 计算拼接位置
            x_start = i * block_width
            x_end = x_start + block.shape[1]
            y_start = j * block_height
            y_end = y_start + block.shape[0]
            
            # 确保不超出边界
            x_end = min(x_end, img_width)
            y_end = min(y_end, img_height)
            
            # 调整block大小以匹配目标区域
            target_height = y_end - y_start
            target_width = x_end - x_start
            
            if block.shape[0] != target_height or block.shape[1] != target_width:
                block = cv2.resize(block, (target_width, target_height))
            
            # 拼接图片块
            merged_img[y_start:y_end, x_start:x_end, :] = block
    
    # 保存拼接后的图片
    cv2.imwrite(output_path, merged_img)
    print(f"图片已拼接完成，保存在: {output_path}")


if __name__ == '__main__':
    # 示例：分割图片
    img_path = 'img/dataset/dis_1 (1).jpg'  # 修改为你的图片路径
    
    # 分割图片为20块
    split_image_20_blocks(img_path, output_dir='img/temp_20')
    
    # 如果需要拼接回去
    # merge_image_20_blocks(input_dir='img/temp_20', 
    #                      output_path='merged_20.jpg',
    #                      original_size=(2560, 1440))


