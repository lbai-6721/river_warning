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

from deeplab import DeeplabV3

if __name__ == "__main__":
    predict_path = 'img/dataset/dis_1 (1).jpg'
    deeplab = DeeplabV3()
    mode = "predict"
    count = False
    name_classes = ["background", "riverbed"]
    video_path = 0
    video_save_path = ""
    video_fps = 25.0
    test_interval = 100
    fps_image_path = "img/test.jpg"
    dir_origin_path = "img/"
    dir_save_path = "img_out/"
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode == "predict":
        # 原始图片处理
        img = cv2.imread(predict_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, dsize=(2560, 1440), interpolation=cv2.INTER_NEAREST)

        if os.path.exists('img/temp'):
            shutil.rmtree('img/temp')
            os.makedirs('img/temp')
        else:
            os.makedirs('img/temp')
        # img -- [height, width, channel]
        # 图片切分
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

        i = j = 0
        for filename in os.listdir('img/temp'):
            image = Image.open(os.path.join('img/temp', filename))
            r_msk = deeplab.detect_image(image, count=count, name_classes=name_classes)
            r_msk = np.array(r_msk)
            msk[208 + j * 512:208 + (j + 1) * 512, i * 512:(i + 1) * 512, :] += r_msk
            i += 1
            if i >= 5:
                j += 1
                i = 0

        # calculate riverbed area
        area = np.where(msk == 224)[0].shape[0]
        print('river area: ' + str(area) + ' / ' + str(img_height * img_width))

        # print(type(msk)) # class<ndarray>
        msk = msk.astype(np.uint8)
        image = cv2.addWeighted(img, 0.5, msk, 0.5, gamma=0)  # 融合图像，获取一组两者在一起的图
        # cv2.imshow('image', image)
        # cv2.waitKey(0)

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
            T_area, T_area_per = Area_detection(msk, (0,0), (1440, 2560))
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
            for i in range(20):
                T_area, T_per = Area_detection(mask, (0, 0+i*128), (1440, 128*(i+1)))
                vetor.append(T_per)
            print(vetor)

        cal_each_area(msk)

        """
            可视化结果：分析两者之间的差异
        
        """

        img = cv2.imread('img/6/172.16.29.10_01_20220901074538377_TIMING_mask.png')

        # 获取所有轮廓
        mask_contours = get_mask_contours(img)

        # 在原始图像上绘制所有轮廓线
        # contours_image = np.zeros(img.shape, dtype=np.uint8)
        # dc = cv2.drawContours(contours_image, mask_contours, -1, (255, 255, 255), cv2.FILLED)

        # 对每个contour块获取上下边沿
        # img_approx = img.copy()
        img_approx = np.zeros(img.shape)
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
            cv2.polylines(img_approx, [up_edge], False, (0, 255, 0), 3)
            cv2.polylines(img_approx, [down_edge], False, (0, 0, 255), 3)

            # 填充河流
            cv2.drawContours(img_approx, np.concatenate(([up_edge], [down_edge]), axis=1), -1, (217, 198, 102),
                             cv2.FILLED)



