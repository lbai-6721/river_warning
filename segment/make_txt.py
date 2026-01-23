import os
import numpy as np
import random

import pandas as pd

data_path = r'G:\项目\堵江识别\deeplab_zhh\deeplab_zhh\image_set\datasets_preprocess\JPEGImages'

data_list = []
for file in os.listdir(data_path):
    name = file.replace('.jpg', '')
    data_list.append(name)

data_list = pd.DataFrame(data_list)
data_list.to_csv(os.path.join('G:\项目\堵江识别\deeplab_zhh\deeplab_zhh\image_set\datasets_preprocess\ImageSets\Segmentation', 'train.txt'), index=None)
# train_list = []
# val_list = []
# i=0
# for data in data_list:
#     if random.randint<0.9:
#