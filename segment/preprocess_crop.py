# -*- coding = utf-8 -*-
# @Time : 2022/11/22 14:30
# @Author : Luxlios
# @File : preprocess_crop.py
# @Software : PyCharm

import os
import cv2
import numpy as np

if __name__ == '__main__':

    img_dir = './image_set/JPEGImages'
    msk_dir = './image_set/SegmentationClass'

    img_save = './image_set/datasets_preprocess/JPEGImages'
    msk_save = './image_set/datasets_preprocess/SegmentationClass'
    if not os.path.exists(img_save):
        os.makedirs(img_save)
    if not os.path.exists(msk_save):
        os.makedirs(msk_save)

    for filename in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, filename),
                         cv2.IMREAD_COLOR)
        msk = cv2.imread(os.path.join(msk_dir, os.path.splitext(filename)[0] + '.png'),
                         cv2.IMREAD_COLOR)
        # print(type(img))
        # break

        # resize to 2560*1440
        img = cv2.resize(img, dsize=(1920, 1152), interpolation=cv2.INTER_NEAREST)
        msk = cv2.resize(msk, dsize=(1920, 1152), interpolation=cv2.INTER_NEAREST)

        # crop
        # img -- [height, width, channel]
        for i in range(3):
            for j in range(1):
                img_temp = img[:, i * 512:(i+1) * 512, :]
                img_temp = img_temp[358 + j * 512:358 + (j+1) * 512, :, :]
                msk_temp = msk[:, i * 512:(i + 1) * 512, :]
                msk_temp = msk_temp[358 + j * 512:358 + (j + 1) * 512, :, :]
                # save
                if img.all() or msk.all() is None:
                    continue
                else:
                    cv2.imwrite(os.path.join(img_save, os.path.splitext(filename)[0] +
                                             '_' + str(j) + str(i) + '.jpg'), img_temp)
                    cv2.imwrite(os.path.join(msk_save, os.path.splitext(filename)[0] +
                                             '_' + str(j) + str(i) + '.png'), msk_temp)

    '''
    msk_dir = r'G:\Desktop\yujing\Step6-Application\deeplabv3-plus-pytorch-main\datasets' \
              r'\SegmentationClass\172.16.29.10_01_20220101104555793_TIMING.png'
    msk = cv2.imread(msk_dir, cv2.IMREAD_COLOR)
    print(np.unique(msk, return_counts=True))
    '''
