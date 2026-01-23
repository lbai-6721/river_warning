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
        image = cv2.addWeighted(img, 0.5, mask, 0.5, gamma=0)    # 在融合的时候设置img和mask权重均为0.5
        cv2.imshow('image', image)                               # gamma是修正系数赋值为0
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
        # 这里发生改变，高斯滤波过程中，调整了高斯函数的标准差sigma
        up_edge = up_edge_list[0]
        g_u = ndimage.gaussian_filter(up_edge[:, 0, 1], sigma=5)
        up_edge[:, 0, 1] = g_u

        # 对下边沿高斯滤波
        down_edge = down_edge_list[0]
        g_d = ndimage.gaussian_filter(down_edge[:, 0, 1], sigma=5)
        down_edge[:, 0, 1] = g_d

        # 画上下边沿
        cv2.polylines(img_approx, [up_edge], False, (0, 255, 0), 2)
        cv2.polylines(img_approx, [down_edge], False, (0, 0, 255), 2)


        # 填充河流
        # cv2.drawContours(img_approx, np.concatenate(([up_edge], [down_edge]), axis=1), -1, color_tuple,
        #                          cv2.FILLED)


    return up_edge, down_edge, img_approx

# img = cv2.imread('img\\172.16.29.10_01_20220901074538377_TIMING_mask.png')
# get_index_visualization(img)

def get_img_mask(predict_path, file_name, mode):
    if mode == "predict":
        img = cv2.imread(predict_path, 0)

        img_height, img_width = img.shape[0], img.shape[1]

        # image = Image.open('./img/night/test.jpg')
        image = Image.open(predict_path)
        msk = deeplab.detect_image(image, count=count, name_classes=name_classes)
        msk = np.array(msk)

        image = np.zeros((img_height, img_width, 3), dtype = np.uint8)
        # 创建一个空的三通道图像
        image[:, :, 0] = img
        image[:, :, 1] = img
        image[:, :, 2] = img
        # 将灰度图像的值映射到每个通道
        img = image

        # calculate riverbed area
        area = np.where(msk == 224)[0].shape[0]
        print('river area(mm^2): ' + str(area) + ' / ' + str(img_height * img_width))

        msk = msk.astype(np.uint8)
        image = cv2.addWeighted(img, 0.5, msk, 0.5, gamma=0)  # 融合图像，获取一组两者在一起的图

    return img, msk, image

from deeplab import DeeplabV3

if __name__ == "__main__":
    # initial settings
    predicted_folder_ = r"./img"
    output_folder = './img_out'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    deeplab = DeeplabV3()
    mode = "predict"
    count = False   # 这里的布尔值来自deeplabV3类中一段统计图像中每个类别的像素数量的代码
    # 这里路径设置为验证图片的存放路径，主要是一些疑似堵江的部分
    name_classes = ["background", "riverbed"]
    test_interval = 100      # 测试间隔来自deeplabV3中一段评估模型处理能力的代码，计算模型的帧率
    # fps_image_path = "img/"
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
            imageing = cv2.imread(os.path.join(predicted_folder, file), cv2.IMREAD_GRAYSCALE)
            imgg = np.zeros((imageing.shape[0], imageing.shape[1], 3), dtype=np.uint8)
            # 创建一个空的三通道图像
            imgg[:, :, 0] = imageing
            imgg[:, :, 1] = imageing
            imgg[:, :, 2] = imageing
            # 将灰度图像的值映射到每个通道
            imageing = imgg
            imageing = cv2.resize(imageing, dsize=(960 * 2, 576 * 2), interpolation=cv2.INTER_NEAREST)

            if pre_mask is None:
                color_tuple = (0, 0, 255)
            else:
                color_tuple = (255, 0, 0)
            print(color_tuple)
            filename = file.replace('.jpg', '')
            print(filename)
            # print('predict image!')
            img, msk, image = get_img_mask(os.path.join(predicted_folder, file), file_name=filename, mode="predict")
            cv2.imwrite(os.path.join(output_folder, name, filename + '_masked.jpg'), image)
            print('get bound info and smooth the edge')
            clear_img = img[:]
            up_edge, down_edge, img_approx = get_index_visualization(msk, img, file, color_tuple=(217, 198, 102))
            print(up_edge)
            print(down_edge)
            sxcha = []
            for sc in up_edge:
                sxcha[sc] = up_edge[sc][1] - down_edge[sc][1]

            print(sxcha)
            cv2.imwrite(os.path.join(output_folder, name, filename + '_curve_mask.jpg'), img_approx)

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
                pre_down = queue_down[1]
                g_d = ndimage.gaussian_filter(pre_down[:, 0, 1], sigma=5)
                pre_down[:, 0, 1] = g_d

                pre_up = queue_up[1]
                g_u = ndimage.gaussian_filter(pre_up[:, 0, 1], sigma=5)
                pre_up[:, 0, 1] = g_u

                now_down = queue_down[0]
                g_d = ndimage.gaussian_filter(now_down[:, 0, 1], sigma=5)
                now_down[:, 0, 1] = g_d

                now_up = queue_up[0]
                g_u = ndimage.gaussian_filter(now_up[:, 0, 1], sigma=5)
                now_up[:, 0, 1] = g_u

                # find the different between area
                A_in_B = cv2.bitwise_and(pre_mask, msk)  # pre ^ now
                A_sub_B = cv2.bitwise_xor(pre_mask, A_in_B)  # pre - pre ^ now
                B_sub_A = cv2.bitwise_xor(msk, A_in_B)  # now - pre ^ now
                cv2.imwrite(os.path.join(output_folder, name, filename + '_A_in_B.jpg'), A_in_B)   # 前后时刻交集
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

                imageing = cv2.resize(imageing, dsize=(960, 576), interpolation=cv2.INTER_NEAREST)
                zeros = np.zeros((imageing.shape), dtype=np.uint8)
                # 原本thickness = -1表示内部填充
                p_mask = cv2.fillPoly(zeros, pre_contours, color=(0, 255, 0))  # Green
                zeros = np.zeros((imageing.shape), dtype=np.uint8)
                n_mask = cv2.fillPoly(zeros, now_contours, color=(255, 0, 0))  # Red
                mask_img = cv2.addWeighted(p_mask, 0.5, n_mask, 0.5, gamma=0)
                # tmp = cv2.addWeighted(pre_mask, 1, msk, 1, gamma=0)
                cv2.imwrite(os.path.join(output_folder, name, filename + '_mask_total.jpg'), mask_img)

                # 第一个参数是轮廓
                # tmp = cv2.drawContours(clear_img, contours, -1, color_map[4], cv2.FILLED)
                total = 0.8 * mask_img + imageing
                cv2.imwrite(os.path.join(output_folder, name, filename + '_total.jpg'), total)

                cv2.polylines(imageing, [pre_down], False, (0, 255, 0), 2)  # BGR
                cv2.polylines(imageing, [now_down], False, (0, 0, 255), 2)  # BGR
                # 因为这里imageing本身是灰度图，所以这里的下边界，一个用的白色，一个用的黑色
                cv2.imwrite(os.path.join(output_folder, name, filename + '_total_curved.jpg'), imageing)

            pre_mask = msk

            print('\n')








