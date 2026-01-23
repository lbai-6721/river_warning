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


if __name__ == '__main__':
    
    # 读取图像
    img = cv2.imread('img\\172.16.29.10_01_20220901074538377_TIMING_mask.png')

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
        cv2.drawContours(img_approx, np.concatenate(([up_edge], [down_edge]), axis=1), -1, (217, 198, 102), cv2.FILLED)


# 显示图像
# plt.imshow(cv2.cvtColor(contours_image, cv2.COLOR_BGR2RGB))
# save_img = cv2.cvtColor(dc, cv2.COLOR_BGR2RGB)
# cv2.imwrite('blur.png', save_img)
# plt.imshow(save_img)
# plt.imshow(cv2.cvtColor(img_approx, cv2.COLOR_BGR2RGB))
# plt.show()

# 保存图像
# cv2.imwrite('res.png', cv2.cvtColor(img_approx, cv2.COLOR_BGR2RGB))
cv2.imwrite('g.png', img_approx)