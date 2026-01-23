# ----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
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
def Area_detection(mask, leftup, rightdown):
    # mask[height, width]
    width = rightdown[0] - leftup[0]
    height = rightdown[1] - leftup[1]
    tgt_area = mask[leftup[0]:leftup[0] + width, leftup[1]:leftup[1] + height, :]
    area = np.where(tgt_area == 224)[0].shape[0]
    area_persent = area / (height * width) * 100

    return area, area_persent

def Alg_3_test():
    # 获取用户输入
    # index_1_x = int(input('Please input the leftup index x:'))
    # index_1_y = int(input('Please input the leftup index y:'))
    # index_2_x = int(input('Please input the rightdown index x:'))
    # index_2_y = int(input('Please input the rightdown index y:'))
    # index_1 = (index_1_x, index_1_y)
    # index_2 = (index_2_x, index_2_y)

    # 选取白石头区域的方法

    # 对应区域计算面积
    # T_area, T_area_per = Area_detection(msk, index_1, index_2)
    T_area, T_area_per = Area_detection(msk, (0, 0), (1440, 2560))
    print('The pixel number is:{}, at {:.2f}% of its target area'.format(T_area, T_area_per))

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
        height = mask_G.shape[0]
        width = mask_G.shape[1]
        previous_flag1 = 0
        previous_flag2 = 0
        current_flag = 0
        upper_index = []
        lower_index = []
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


def cal_each_area(mask):
    vetor = []
    for i in range(19):
        T_area, T_per = Area_detection(mask, (0, 0 + i * 128), (1440, 128 * (i + 1)))
        vetor.append(T_per)
# cal_each_area(msk)

# 获取边界信息 get_index_visualization 提取河流上下边界
def get_index_visualization(img, image, file_name, color_tuple):
    # img: ndarray  file_name: str
    # 获取所有轮廓
    mask_contours = get_mask_contours(img)

    # img_approx = np.zeros(img.shape)
    img_approx = image
    approx_list = []
    up_edge_list = []
    down_edge_list = []
    for contour in mask_contours:
        # 最左和最右点的index
        leftmost_index = contour[:, :, 0].argmin()
        rightmost_index = contour[:, :, 0].argmax()

        # 上下边沿所有点的列表
        up_edge_list.append(np.concatenate((contour[rightmost_index + 1:], contour[0:leftmost_index]), axis=0))
        down_edge_list.append(contour[leftmost_index:rightmost_index + 1])

        # 对上边沿高斯滤波
        up_edge = up_edge_list[0]
        g_u = ndimage.gaussian_filter(up_edge[:, 0, 1], sigma=30)
        up_edge[:, 0, 1] = g_u

        # 对下边沿高斯滤波
        down_edge = down_edge_list[0]
        g_d = ndimage.gaussian_filter(down_edge[:, 0, 1], sigma=30)
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

# 获取掩码 get_img_mask进行语义分割
def get_img_mask(predict_path, file_name, mode):
    if mode == "predict":
        img = cv2.imread(predict_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, dsize=(2560, 1440), interpolation=cv2.INTER_NEAREST)

        if os.path.exists('img/temp'):
            shutil.rmtree('img/temp')
            os.makedirs('img/temp')
        else:
            os.makedirs('img/temp')
        # img -- [height, width, channel]
        for i in range(5):   #分割图像为5*2的小图像
            for j in range(2):   #每个小图像再分割为2*1的小图像
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
                r_msk = deeplab.detect_image(image, count=count, name_classes=name_classes)  #调用deeplab模型进行语义分割
                r_msk = np.array(r_msk)
                msk[208 + j * 512:208 + (j + 1) * 512, i * 512:(i + 1) * 512, :] += r_msk  #将分割结果拼接起来

        # calculate riverbed area
        area = np.where(msk == 224)[0].shape[0]  #计算河体面积
        print('river area: ' + str(area) + ' / ' + str(img_height * img_width))

        msk = msk.astype(np.uint8)
        image = cv2.addWeighted(img, 0.5, msk, 0.5, gamma=0)  # 融合图像，获取一组两者在一起的图
    return img, msk, image

from deeplab import DeeplabV3

if __name__ == "__main__":
    # initial settings
    predicted_folder_ = r'/hy-tmp/data_pairs_normal2dis/data_pairs_normal'   #绝对路径
    output_folder = r'/hy-tmp/data_pairs_output_normal'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # initial the model
    # if changing the backbone in DeeplabV3, find ./net/deeplabv3_plus.py to change
    deeplab = DeeplabV3()
    mode = "predict"
    count = False
    name_classes = ["background", "riverbed"]
    test_interval = 100
    fps_image_path = "img/test.jpg"
    dir_origin_path = "img/"
    dir_save_path = "img_out/"
    simplify = True
    onnx_save_path = "model_data/models.onnx"
    color_map = [
        [0, 255, 255],  # Yellow           --0   175 238 238
        [0, 192, 0],  # Green            --1
        [0, 0, 255],  # Red              --2
        [0, 165, 255],  # Orange           --3
        [128, 0, 128],  # Blue           --4

    ]
    for predicted_folder in os.listdir(predicted_folder_):
        print(predicted_folder)
        queue_up, queue_down, queue_image = [], [], []
        pre_mask = None
        name = predicted_folder
        if not os.path.exists(os.path.join(output_folder, name)):
            os.makedirs(os.path.join(output_folder, name))
        predicted_folder = os.path.join(predicted_folder_, predicted_folder)
        for file in os.listdir(predicted_folder):
            # 读取图片
            image_path = os.path.join(predicted_folder, file)
            imageing = cv2.imread(image_path, cv2.IMREAD_COLOR)
            
            # 检查图像是否成功读取
            if imageing is None:
                print(f"警告: 无法读取图像文件 {image_path}，跳过该文件")
                continue
            
            imageing = cv2.resize(imageing, dsize=(2560, 1440), interpolation=cv2.INTER_NEAREST)
            # 设置颜色
            if pre_mask is None:
                color_tuple = (0, 0, 255)
            else:
                color_tuple = (255, 0, 0)
            print(color_tuple)
            filename = file.replace('.jpg', '')
            print(filename)
            print('predict image!')
            # 获取掩码 get_img_mask 进行语义分割
            img, msk, image = get_img_mask(os.path.join(predicted_folder, file), file_name=filename, mode="predict")
            cv2.imwrite(os.path.join(output_folder, name, filename + '_masked.jpg'), image)  #保存带掩码的融合图像
            # 获取边界信息
            print('get bound info and smooth the edge')
            clear_img = img[:]
            # 保存带掩码的融合图像 get_index_visualization 提取河流上下边界
            up_edge, down_edge, img_approx = get_index_visualization(msk, img, file, color_tuple=(217, 198, 102))
            cv2.imwrite(os.path.join(output_folder, name, filename + '_curve_mask.jpg'), img_approx)

            # 队列：使用队列存储前一时刻的信息，最多维护三帧，用于时序对比
            if len(queue_up) > 3:
                queue_up.pop(-1)
            if len(queue_down) > 3:
                queue_down.pop(-1)
            # 使用队列存储前一时刻的信息
            queue_up.append(up_edge)
            queue_down.append(down_edge)
            # 进行前后差异对比
            if pre_mask is not None:
                print('Start compare!')
                # fix the down and up edge
                # 对前一帧的下边界和上边界进行高斯平滑处理
                pre_down = queue_down[1]
                g_d = ndimage.gaussian_filter(pre_down[:, 0, 1], sigma=30)
                pre_down[:, 0, 1] = g_d

                pre_up = queue_up[1]
                g_u = ndimage.gaussian_filter(pre_up[:, 0, 1], sigma=30)
                pre_up[:, 0, 1] = g_u

                # 对后一帧的下边界和上边界进行高斯平滑处理
                now_down = queue_down[0]
                g_d = ndimage.gaussian_filter(now_down[:, 0, 1], sigma=30)
                now_down[:, 0, 1] = g_d

                now_up = queue_up[0]
                g_u = ndimage.gaussian_filter(now_up[:, 0, 1], sigma=30)
                now_up[:, 0, 1] = g_u

                # find the different between area
                # 计算前后帧重叠区域
                A_in_B = cv2.bitwise_and(pre_mask, msk)  # pre ^ now
                A_sub_B = cv2.bitwise_xor(pre_mask, A_in_B)  # pre - pre ^ now
                B_sub_A = cv2.bitwise_xor(msk, A_in_B)  # now - pre ^ now
                cv2.imwrite(os.path.join(output_folder, name, filename + '_A_in_B.jpg'), A_in_B) # 前后帧重叠区域
                cv2.imwrite(os.path.join(output_folder, name, filename + '_A_sub_B.jpg'), A_sub_B) # 前帧独有区域（可能减少）
                cv2.imwrite(os.path.join(output_folder, name, filename + '_B_sub_A.jpg'), B_sub_A) # 后帧独有区域（可能增加）

                # my way find the different area
                # print(len(pre_down)==len(pre_up))
                # 获取前帧的轮廓
                pre_mask = cv2.cvtColor(pre_mask, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(pre_mask, 0, 255, 0)
                pre_contours, im = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 第一个参数是轮廓
                # tmp = cv2.drawContours(msk, contours, -1, color_map[3], cv2.FILLED)

                # 获取后帧的轮廓
                mask = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(mask, 0, 255, 0)
                now_contours, im = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                zeros = np.zeros((imageing.shape), dtype=np.uint8)
                # 原本thickness = -1表示内部填充，填充前帧的轮廓
                p_mask = cv2.fillPoly(zeros, pre_contours, color=(255, 0, 0))  # blue蓝色（前帧轮廓）
                zeros = np.zeros((imageing.shape), dtype=np.uint8)
                # 填充后帧的轮廓
                n_mask = cv2.fillPoly(zeros, now_contours, color=(0, 0, 255))  # red红色（当前帧轮廓）
                mask_img = cv2.addWeighted(p_mask, 0.5, n_mask, 0.5, gamma=0)
                # tmp = cv2.addWeighted(pre_mask, 1, msk, 1, gamma=0)
                cv2.imwrite(os.path.join(output_folder, name, filename + '_mask_total.jpg'), mask_img)  #保存融合图像

                # 第一个参数是轮廓
                # tmp = cv2.drawContours(clear_img, contours, -1, color_map[4], cv2.FILLED)
                total = 0.5 * mask_img + imageing
                cv2.imwrite(os.path.join(output_folder, name, filename + '_total.jpg'), total)  #保存mask和原始  融合图像

                cv2.polylines(imageing, [pre_down], False, (255, 0, 0), 2)  # BGR
                cv2.polylines(imageing, [now_down], False, (0, 0, 255), 2)  # BGR
                cv2.imwrite(os.path.join(output_folder, name, filename + '_total_curved.jpg'), imageing) #下边界绘制图像

            pre_mask = msk

            print('\n')








