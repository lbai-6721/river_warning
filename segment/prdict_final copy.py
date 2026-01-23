# ============================================================#
#   河堵预警算法 - 最终版本 (支持白天和夜间模式)
#   
#   功能说明：
#   - 支持批量图像处理
#   - 语义分割识别河床区域
#   - 提取河流上下边界
#   - 前后帧对比分析
#   - 完善的错误处理机制
#
#   场景模式：
#   1. 夜间模式 (night)：
#      - 灰度图像处理
#      - 整图语义分割（不切分）
#      - 弱高斯平滑 (sigma=5)
#      - 分辨率：1920×1152
#      - 适用：低光照、夜间拍摄图像
#
#   2. 白天模式 (day)：
#      - 彩色图像处理
#      - 切分处理（5×2小块，每块512×512）
#      - 强高斯平滑 (sigma=30)
#      - 分辨率：2560×1440
#      - 适用：正常光照、白天拍摄图像
#
#   使用方法：
#   1. 修改第314行的 SCENE_MODE 参数：'night' 或 'day'
#   2. 修改第317行的 predicted_folder_ 为输入目录
#   3. 修改第318行的 output_folder 为输出目录
#   4. 运行脚本
#
#   夜间目录：
#   predicted_folder_ = r"/hy-tmp/Nightdata_pairs_normal2dis/data_pairs_normal"
#   output_folder = r'/hy-tmp/Nightdata_pairs_output_normal2dis'
#
#   白天目录：
#    predicted_folder_ = r'/hy-tmp/data_pairs_normal2dis/data_pairs_normal'
#    output_folder = r'/hy-tmp/data_pairs_output_normal'
# ============================================================#
import time
import os
import shutil
import cv2
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def storeimage(img):
    print('storing image....')

def get_mask_contours(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化图像，将所有非零像素设为1
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_filter = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)

        # 处理掉小的轮廓区域，这个区域的大小自己定义。
        if area > 50 * 50:
            contours_filter.append(cnt)

    return contours_filter

"""
    算法3 
    获取输入的坐标区域，获得该区域的水体面积和百分比
    Area_detection(mask, leftup, rightdown):
"""
# def Area_detection(mask, leftup, rightdown):
#     # mask[height, width]
#     width = rightdown[0] - leftup[0]
#     height = rightdown[1] - leftup[1]
#     tgt_area = mask[leftup[0]:leftup[0] + width, leftup[1]:leftup[1] + height, :]
#     area = np.where(tgt_area == 224)[0].shape[0]
#     area_persent = area / (height * width) * 100
#
#     return area, area_persent

# def Alg_3_test():
#     # 获取用户输入
#     # index_1_x = int(input('Please input the leftup index x:'))
#     # index_1_y = int(input('Please input the leftup index y:'))
#     # index_2_x = int(input('Please input the rightdown index x:'))
#     # index_2_y = int(input('Please input the rightdown index y:'))
#     # index_1 = (index_1_x, index_1_y)
#     # index_2 = (index_2_x, index_2_y)
#
#     # 选取白石头区域的方法
#
#     # 对应区域计算面积
#     # T_area, T_area_per = Area_detection(msk, index_1, index_2)
#     T_area, T_area_per = Area_detection(msk, (0, 0), (1440, 2560))
#     print('The pixel number is:{}, at {:.2f}% of its target area'.format(T_area, T_area_per))

# Alg_3_test()


"""
    算法4：获取河体上下沿坐标
    def get_up_and_down_index(msk)
    输入图像的二值化
"""
def get_color_channels(img):
    return img[:, :, 0], img[:, :, 1], img[:, :, 2]

def get_up_and_down_index(mask, img):
        # 二值化处理
        R, G, B = get_color_channels(mask)
        mask_G = np.array(R * 299 / 1000 + G * 587 / 1000 + B * 114 / 1000)  # 转化为灰度图
        mask_G[mask_G > 0] = 1  # 转化为0，1矩阵

        # import pdb; pdb.set_trace()
        # 单通道图进行遍历搜索
        # 取mask的shape宽高
        height = mask_G.shape[0]
        width = mask_G.shape[1]
        previous_flag1 = 0
        previous_flag2 = 0
        current_flag = 0
        upper_index = []
        lower_index = []
        # 按列进行遍历
        for i in range(width):
            for j in range(height):
                # 这里用嵌套循环的方法来搜索边界信息
                if mask_G[j][i] == 1:
                    current_flag = 1
                else:
                    current_flag = 0

                if previous_flag1 == 0 and current_flag == 1:
                    # 从河岸向河体转
                    upper_index.append([j, i])
                    # import pdb; pdb.set_trace()
                elif previous_flag1 == 1 and current_flag == 0:
                    # 由河体向河岸转移
                    lower_index.append([j, i])

                # 标签转移
                previous_flag2 = previous_flag1
                previous_flag1 = current_flag

        # 更换矩阵中对应部分的值的颜色
        # 对于河体上沿，赋予蓝色
        # 对于河体下沿，赋予红色
        # mask[mask>0] = 0
        for index in upper_index:
            mask[index[0]][index[1]] = [255, 0, 0]
            mask_G[index[0]][index[1]] = 255
        for index in lower_index:
            mask[index[0]][index[1]] = [0, 0, 255]
            mask_G[index[0]][index[1]] = 255

        mask_G = mask_G.astype(np.uint8)
        image = cv2.addWeighted(img, 0.5, mask, 0.5, gamma=0)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        return upper_index, lower_index, mask_G

# 算法4测试
# get_up_and_down_index(msk, img)
"""
算法5：比较两个时刻的水体面积并进行对比

"""


# def cal_each_area(mask):
#     vetor = []
#     for i in range(19):
#         T_area, T_per = Area_detection(mask, (0, 0 + i * 128), (1440, 128 * (i + 1)))
#         vetor.append(T_per)
# # cal_each_area(msk)

def get_index_visualization(img, image, file_name, color_tuple, sigma=5):
    """
    提取河流边界并可视化
    sigma: 高斯滤波标准差，夜间用5（弱平滑），白天用30（强平滑）
    """
    # img: ndarray  file_name: str
    # 获取所有轮廓
    mask_contours = get_mask_contours(img)

    # img_approx = np.zeros(img.shape)
    img_approx = image
    approx_list = []
    up_edge_list = []
    down_edge_list = []
    
    # 初始化 up_edge 和 down_edge，防止轮廓为空时未定义
    up_edge = np.array([], dtype=np.int32).reshape(0, 1, 2)
    down_edge = np.array([], dtype=np.int32).reshape(0, 1, 2)
    
    # 检查是否有轮廓
    if len(mask_contours) == 0:
        print("警告: 未找到轮廓（可能是语义分割结果为空），返回空边界")
        return up_edge, down_edge, img_approx
    
    for contour in mask_contours:
        # 最左和最右点的index
        leftmost_index = contour[:, :, 0].argmin()
        rightmost_index = contour[:, :, 0].argmax()

        # 上下边沿所有点的列表
        up_edge_list.append(np.concatenate((contour[rightmost_index + 1:], contour[0:leftmost_index]), axis=0))
        down_edge_list.append(contour[leftmost_index:rightmost_index + 1])

        # 对上边沿高斯滤波
        up_edge = up_edge_list[0]
        g_u = ndimage.gaussian_filter(up_edge[:, 0, 1], sigma=sigma)
        up_edge[:, 0, 1] = g_u

        # 对下边沿高斯滤波
        down_edge = down_edge_list[0]
        g_d = ndimage.gaussian_filter(down_edge[:, 0, 1], sigma=sigma)
        down_edge[:, 0, 1] = g_d

        # 画上下边沿
        cv2.polylines(img_approx, [up_edge], False, (0, 255, 0), 2)
        cv2.polylines(img_approx, [down_edge], False, (0, 0, 255), 2)

        # 填充河流
        # cv2.drawContours(img_approx, np.concatenate(([up_edge], [down_edge]), axis=1), -1, color_tuple,
        #                          cv2.FILLED)


    return up_edge, down_edge, img_approx
#
# img = cv2.imread('img\\172.16.29.10_01_20220901074538377_TIMING_mask.png')
# get_index_visualization(img)

def get_img_mask_night(predict_path, file_name, mode):
    """夜间模式：灰度图，整图处理"""
    if mode == "predict":
        img = cv2.imread(predict_path, 0)
        
        if img is None:
            raise ValueError(f"无法读取图像文件: {predict_path}")

        img_height, img_width = img.shape[0], img.shape[1]

        # image = Image.open('./img/night/test.jpg')
        image = Image.open(predict_path)
        msk = segNet.detect_image(image, count=count, name_classes=name_classes)
        msk = np.array(msk)

        image = np.zeros((img_height, img_width, 3), dtype = np.uint8)
        # 创建一个空的三通道图像
        image[:, :, 0] = img
        image[:, :, 1] = img
        image[:, :, 2] = img
        # 将灰度图像的值赋给每个通道
        img = image

        # calculate riverbed area
        area = np.where(msk == 224)[0].shape[0]
        pui=area/(img_height * img_width)*100
        print('百分比为：'+str(pui))
        print('river area(mm^2): ' + str(area) + ' / ' + str(img_height * img_width))

        msk = msk.astype(np.uint8)
        image = cv2.addWeighted(img, 0.5, msk, 0.5, gamma=0)  # 融合图像，获取一组两者在一起的图

    return img, msk, image

def get_img_mask_day(predict_path, file_name, mode):
    """白天模式：彩色图，切分处理（高分辨率）"""
    if mode == "predict":
        img = cv2.imread(predict_path, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError(f"无法读取图像文件: {predict_path}")
            
        img = cv2.resize(img, dsize=(2560, 1440), interpolation=cv2.INTER_NEAREST)

        if os.path.exists('img/temp'):
            shutil.rmtree('img/temp')
            os.makedirs('img/temp')
        else:
            os.makedirs('img/temp')
        
        # img -- [height, width, channel]
        # 切分图像为5*2的小图像
        for i in range(5):
            for j in range(2):
                img_temp = img[:, i * 512:(i + 1) * 512, :]
                img_temp = img_temp[208 + j * 512:208 + (j + 1) * 512, :, :]
                # save
                if img_temp is None:
                    continue
                else:
                    cv2.imwrite(os.path.join('img/temp', str(j) + str(i) + '.jpg'), img_temp)

        img_height, img_width = img.shape[0], img.shape[1]
        msk = np.zeros([img_height, img_width, 3])

        # 修复：按照切分时的顺序进行拼接，而不是依赖 os.listdir 的不确定顺序
        for i in range(5):   # 水平方向：0,1,2,3,4
            for j in range(2):   # 垂直方向：0,1
                filename = str(j) + str(i) + '.jpg'  # 文件名: 00.jpg, 10.jpg, 01.jpg, ...
                image_path = os.path.join('img/temp', filename)
                if not os.path.exists(image_path):
                    print(f"警告: 切分图像 {filename} 不存在，跳过")
                    continue
                image = Image.open(image_path)
                r_msk = segNet.detect_image(image, count=count, name_classes=name_classes)
                r_msk = np.array(r_msk)
                msk[208 + j * 512:208 + (j + 1) * 512, i * 512:(i + 1) * 512, :] += r_msk

        # calculate riverbed area
        area = np.where(msk == 224)[0].shape[0]
        pui = area / (img_height * img_width) * 100
        print('百分比为：' + str(pui))
        print('river area: ' + str(area) + ' / ' + str(img_height * img_width))

        msk = msk.astype(np.uint8)
        image = cv2.addWeighted(img, 0.5, msk, 0.5, gamma=0)  # 融合图像，获取一组两者在一起的图

    return img, msk, image

def get_img_mask(predict_path, file_name, mode, scene_mode='night'):
    """
    统一接口：根据场景模式选择处理方式
    scene_mode: 'night' 或 'day'
    """
    if scene_mode == 'day':
        return get_img_mask_day(predict_path, file_name, mode)
    else:
        return get_img_mask_night(predict_path, file_name, mode)

from deeplab import DeeplabV3
from SSSegnet import SegNet

if __name__ == "__main__":

    # ========== 场景模式配置 ==========
    # 'night': 夜间模式 - 灰度图，整图处理，弱平滑(sigma=5)，分辨率1920×1152
    # 'day':   白天模式 - 彩色图，切分处理，强平滑(sigma=30)，分辨率2560×1440
    SCENE_MODE = 'night'  # 修改此参数切换模式：'night' 或 'day'
    
    # initial settings
    predicted_folder_ =r"/hy-tmp/Nightdata_pairs_normal2dis/data_pairs_normal"
    output_folder = '/hy-tmp/Nightdata_pairs_output_normal2dis'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    segNet = SegNet()
    mode = "predict"
    count = False   # 布尔值的用途？
    name_classes = ["background", "riverbed"]
    test_interval = 100      # 测试间隔的用途？
    #fps_image_path = "img/test"
    dir_origin_path = "img/test"
    dir_save_path = "img/img_out/"
    simplify = True
    onnx_save_path = "model_data/models.onnx"
    
    # 根据场景模式设置参数
    if SCENE_MODE == 'day':
        sigma_value = 30  # 白天：强平滑
        resize_dim = (2560, 1440)  # 白天：高分辨率
        print("=" * 60)
        print("场景模式：白天 (彩色图像，切分处理)")
        print("=" * 60)
    else:
        sigma_value = 5   # 夜间：弱平滑
        resize_dim = (1920, 1152)  # 夜间：中分辨率
        print("=" * 60)
        print("场景模式：夜间 (灰度图像，整图处理)")
        print("=" * 60)
    color_map = [
        [0, 255, 255],  # Yellow           --0   175 238 238
        [0, 192, 0],  # Green            --1
        [0, 0, 255],  # Red              --2
        [0, 165, 255],  # Orange           --3
        [128, 0, 128],  # Blue           --4

    ]     # 设置了一下mask的颜色
    for predicted_folder in os.listdir(predicted_folder_):

        print(predicted_folder)
        queue_up, queue_down, queue_image = [], [], []
        pre_mask = None
        name = predicted_folder
        if not os.path.exists(os.path.join(output_folder, name)):
            os.makedirs(os.path.join(output_folder, name))
        predicted_folder = os.path.join(predicted_folder_, predicted_folder)
        for file in os.listdir(predicted_folder):
            # 跳过目录和隐藏文件
            file_path = os.path.join(predicted_folder, file)
            if os.path.isdir(file_path) or file.startswith('.'):
                continue
            
            # 只处理图像文件
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                print(f"跳过非图像文件: {file}")
                continue
            
            # 根据场景模式读取图像
            if SCENE_MODE == 'day':
                imageing = cv2.imread(file_path, cv2.IMREAD_COLOR)  # 白天：彩色
            else:
                imageing = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # 夜间：灰度
            
            if imageing is None:
                print(f"警告: 无法读取图像文件 {file_path}，跳过")
                continue
            
            # 根据场景模式调整尺寸
            imageing = cv2.resize(imageing, dsize=resize_dim, interpolation=cv2.INTER_NEAREST)

            if pre_mask is None:
                color_tuple = (0, 0, 255)
            else:
                color_tuple = (255, 0, 0)
            print(color_tuple)
            filename = file.replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('.bmp', '')
            print(filename)
            
            try:
                print(f'predict image! (模式: {SCENE_MODE})')
                img, msk, image = get_img_mask(file_path, file_name=filename, mode="predict", scene_mode=SCENE_MODE)
                cv2.imwrite(os.path.join(output_folder, name, filename + '_masked.jpg'), image)
                print('get bound info and smooth the edge')
                clear_img = img[:]
                up_edge, down_edge, img_approx = get_index_visualization(msk, img, file, color_tuple=(217, 198, 102), sigma=sigma_value)
                cv2.imwrite(os.path.join(output_folder, name, filename + '_curve_mask.jpg'), img_approx)

                # 检查边界是否为空（语义分割未检测到河床）
                if len(up_edge) == 0 or len(down_edge) == 0:
                    print(f"警告: 图像 {filename} 未检测到河床边界，跳过后续处理")
                    pre_mask = msk  # 仍然保存掩码用于下一次对比
                    print('\n')
                    continue

                # if len(queue_up) > 3:
                #     queue_up.pop(-1)
                # if len(queue_down) > 3:
                #     queue_down.pop(-1)
                # 使用队列存储前一时刻的信息
                queue_up.append(up_edge)
                queue_down.append(down_edge)
                # 进行前后差异对比
                if pre_mask is not None and len(queue_up) >= 2 and len(queue_down) >= 2:
                    print('Start compare!')
                    # fix the down and up edge
                    pre_down = queue_down[0]
                    g_d = ndimage.gaussian_filter(pre_down[:, 0, 1], sigma=sigma_value)
                    pre_down[:, 0, 1] = g_d

                    pre_up = queue_up[0]
                    g_u = ndimage.gaussian_filter(pre_up[:, 0, 1], sigma=sigma_value)
                    pre_up[:, 0, 1] = g_u

                    now_down = queue_down[1]
                    g_d = ndimage.gaussian_filter(now_down[:, 0, 1], sigma=sigma_value)
                    now_down[:, 0, 1] = g_d

                    now_up = queue_up[1]
                    g_u = ndimage.gaussian_filter(now_up[:, 0, 1], sigma=sigma_value)
                    now_up[:, 0, 1] = g_u

                    # find the different between area
                    A_in_B = cv2.bitwise_and(pre_mask, msk)  # pre ^ now
                    A_sub_B = cv2.bitwise_xor(pre_mask, A_in_B)  # pre - pre ^ now
                    B_sub_A = cv2.bitwise_xor(msk, A_in_B)  # now - pre ^ now
                    cv2.imwrite(os.path.join(output_folder, name, filename + '_A_in_B.jpg'), A_in_B)
                    cv2.imwrite(os.path.join(output_folder, name, filename + '_A_sub_B.jpg'), A_sub_B)
                    cv2.imwrite(os.path.join(output_folder, name, filename + '_B_sub_A.jpg'), B_sub_A)

                    # my way find the different area
                    # print(len(pre_down)==len(pre_up))
                    pre_mask = cv2.cvtColor(pre_mask, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(pre_mask, 0, 255, 0)
                    pre_contours, im = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 第一个参数是轮廓
                    # tmp = cv2.drawContours(msk, contours, -1, color_map[3], cv2.FILLED)

                    mask = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(mask, 0, 255, 0)
                    now_contours, im = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # 根据场景模式设置可视化尺寸
                    if SCENE_MODE == 'day':
                        vis_size = (1280, 720)  # 白天：降采样用于可视化
                    else:
                        vis_size = (960, 576)   # 夜间：默认尺寸
                    imageing = cv2.resize(imageing, dsize=vis_size, interpolation=cv2.INTER_NEAREST)
                    zeros = np.zeros((imageing.shape), dtype=np.uint8)
                    # 原本thickness = -1表示内部填充
                    p_mask = cv2.fillPoly(zeros, pre_contours, color=(255, 255, 0))  # blue
                    zeros = np.zeros((imageing.shape), dtype=np.uint8)
                    n_mask = cv2.fillPoly(zeros, now_contours, color=(0, 255, 255))  # red
                    mask_img = cv2.addWeighted(p_mask, 0.5, n_mask, 0.5, gamma=0)
                    # tmp = cv2.addWeighted(pre_mask, 1, msk, 1, gamma=0)
                    cv2.imwrite(os.path.join(output_folder, name, filename + '_mask_total.jpg'), mask_img)

                    # 第一个参数是轮廓
                    # tmp = cv2.drawContours(clear_img, contours, -1, color_map[4], cv2.FILLED)
                    total = 0.5 * mask_img + imageing
                    cv2.imwrite(os.path.join(output_folder, name, filename + '_total.jpg'), total)

                    cv2.polylines(imageing, [pre_down], False, (255, 255, 0), 2)  # BGR
                    cv2.polylines(imageing, [now_down], False, (0, 255, 255), 2)  # BGR
                    cv2.imwrite(os.path.join(output_folder, name, filename + '_total_curved.jpg'), imageing)

                pre_mask = msk

            except Exception as e:
                print(f"错误: 处理文件 {file} 时出错: {e}")
                import traceback
                traceback.print_exc()
            
            print('\n')








