# This repo is used to convert the json file to .jpg & .png format files
import base64
import json
import os
import os.path as osp

import PIL.Image
import numpy as np
from labelme import utils

'''
我使用的labelme版本是3.16.7，建议使用该版本的labelme，有些版本的labelme会发生错误
此处生成的标签图是8位彩色图，每个像素点的值就是这个像素点所属的种类
'''

if __name__ == '__main__':
    jpgs_path = "./image_set/JPEGImages"
    pngs_path = "./image_set/SegmentationClass"
    classes = ["_background_", "river", "通畅", "堵塞", "畅通", "没堵", "堵", "ybl", "river1", "river2", "river3", "river4", "river5", "river6", "river7", "river8", "river9", "river10", "river11", "river12", "river13", "river14", "river15", "river16", "river17", "river18", "river19", "river20", "江体", "20220921000033", "不堵"]
    if not os.path.exists(jpgs_path):
        os.makedirs(jpgs_path)
    if not os.path.exists(pngs_path):
        os.makedirs(pngs_path)

    # count = os.listdir(r"G:\项目\堵江识别\deeplab_zhh\deeplab_zhh\image_set\disease")
    count = os.listdir(r"D:\BaiduNetdiskDownload\语义分割部分\语义分割部分\image_set\disease")
    # import pdb;
    # pdb.set_trace()
    for i in range(0, len(count)):
        path = os.path.join("./image_set/disease", count[i])
        # import pdb;
        # pdb.set_trace()
        if os.path.isfile(path) and path.endswith('json'):  # 如果是json文件，那就打开它
            data = json.load(open(path, encoding="utf-8"))  # 用json.load打开json文件

            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')

            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {'_background_': 0, 'river': 1}
            for shape in data['shapes']:  # data就是image和json文件
                label_name = shape['label']
                # for i1 in range(len(classes)):
                    # label_name_to_value[classes[i1]] = 1

                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]

                else:
                    label_name_to_value[label_name] = 1
                    # label_value = label_name_to_value[label_name]
                    # label_value = len(label_name_to_value)  # 这里命名的逻辑是通过label_value的长度来进行赋值
                    # label_name_to_value[label_name] = label_value
                # 把当前标签转化为一个统一的值，labelname, lable_value

            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            # assert label_values == list(range(len(label_values)))
            # 判断label_values中是否每个元素都有赋值

            lbl, lbl_name = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
            # lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

            PIL.Image.fromarray(img).save(osp.join(jpgs_path, os.path.splitext(count[i])[0] + '.jpg'))

            new = np.zeros([np.shape(img)[0], np.shape(img)[1]])
            for name in label_names:
                index_json = label_names.index(name)
                index_all = classes.index(name)
                new = new + index_all * (np.array(lbl) == index_json)

            utils.lblsave(osp.join(pngs_path, os.path.splitext(count[i])[0] + '.png'), new)
            print('Saved ' + os.path.splitext(count[i])[0] + '.jpg and ' + os.path.splitext(count[i])[0] + '.png')
