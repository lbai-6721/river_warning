import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import pandas as pd
from net import CNN
import os
from sklearn.metrics import precision_score, recall_score, f1_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# 创建数据集类
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datax, data_label):
        self.data1 = datax
        self.label = data_label

    def __getitem__(self, indx):
        # 返回一个数据样本
        return self.data1[indx], self.label[indx]
        # pass

    def __len__(self):
        # 返回数据集的大小
        return len(self.data1)


# 自定义函数，判断是否为奇数行
def skip_odd_rows(row):
    return row % 2 == 1

# 读取CSV文件，跳过奇数行
# up差
up = pd.read_csv(r'./up.csv', skiprows=lambda x: skip_odd_rows(x), encoding='unicode_escape')
up = np.array(up)
up = up[:, 1:]
# 打印读取的数据
n = len(up)
# down差
down = pd.read_csv(r'./down.csv', skiprows=lambda x: skip_odd_rows(x), encoding='unicode_escape')
down = np.array(down)
down = down[:, 1:]
up = pd.DataFrame(up)
down = pd.DataFrame(down)
# 数据结合
data1 = pd.concat([up, down], axis=1)
data1 = np.array(data1)
data1 = data1.astype(float)  # numpy强制类型转换
data1 = torch.tensor(data1, dtype=torch.float32)
data1 = data1.reshape(n, 2, 20)
# print(data1)
# 河体面积
# data2=[]
# data_label=[]
area = pd.read_csv(r'./area.csv', encoding='unicode_escape')
area = np.array(area)
data2 = area[0:251, 1:11]
data2 = data2.astype(float)  # numpy强制类型转换
data2 = torch.tensor(data2, dtype=torch.float32)
# label
data_label = area[0:251, 11]
data_label = data_label.astype(float)  # numpy强制类型转换
data_label = torch.tensor(data_label, dtype=torch.float32)

data = []
for i in range(len(data1)):
    data.append([data1[i], data2[i]])

# 参数
BATCH_SIZE = 20
device = torch.device('cpu')

# 数据
data_test = MyDataset(data, data_label)
dataloader_test = Data.DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False)
loss_func = nn.CrossEntropyLoss()  # 定义损失函数

print("testing")
model = CNN()
model.load_state_dict(torch.load('./checkpoint.pth'))
with torch.no_grad():
    model.eval()
    accuracy = []
    recall = []
    PRES = []
    for i, (data, batch_label) in enumerate(dataloader_test):
        batch_x = data[0].to(device) #up和down差值
        batch_y = data[1].to(device)  #area面积
        print("Batch X shape:", batch_x.shape)
        print("Batch Y shape:", batch_y.shape)

        pred_label = model(batch_x, batch_y) #0：没灾害 1：有灾害
        loss = loss_func(pred_label, batch_label.long())
        total = batch_label.size(0)
        pred_label = torch.argmax(pred_label, dim=-1)
        correct = (pred_label == batch_label).sum().item()
        accuracy.append(correct / total)
        recall.append(recall_score(batch_label, pred_label))
        PRES.append(precision_score(batch_label, pred_label))

        print("TEST {}/{} || Loss {:.2f}, ACC: {:.2f} pred {:} Target {:}"
                .format(i, len(dataloader_test), loss.item(), np.mean(accuracy), pred_label, batch_label))

print("TEST SET ACC:{:.2f} Recall {:.2f} PRE {:.2f}".format(np.mean(accuracy), np.mean(recall), np.mean(PRES)))