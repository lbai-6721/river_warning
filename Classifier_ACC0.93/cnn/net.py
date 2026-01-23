import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(self,
                 x_channel=2,
                 y_channel=1,
                 x_length=20,
                 y_length=10,
                 shared_channel=20,
                 out_class=2
                 ):
        super(CNN, self).__init__()

        self.in_conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, stride=1)

        self.fc_x = nn.Sequential(
            nn.Linear(x_length, x_length),  nn.Dropout(0.1),
            nn.Linear(x_length, x_length),  nn.Dropout(0.1),
            nn.Linear(x_length, shared_channel)
        )

        self.fc_y = nn.Sequential(
            nn.Linear(y_length, y_length),  nn.Dropout(0.1),
            nn.Linear(y_length, y_length),  nn.Dropout(0.1),
            nn.Linear(y_length, shared_channel)
        )
        self.concat_channel = shared_channel + shared_channel

        self.fc = nn.Sequential(
            nn.Linear(self.concat_channel, self.concat_channel // 2), nn.ReLU(inplace=True),
            nn.Linear(self.concat_channel // 2, self.concat_channel // 4), nn.ReLU(inplace=True),
            nn.Linear(self.concat_channel // 4, out_class)
        )

    # two input
    def forward(self, x, y):
        # import pdb; pdb.set_trace()
        bs = x.shape[0]
        x = self.in_conv(x).mean(1) #[batch_size, 2, 20] -> [batch_size, 1, 20] -> [batch_size, 20]
        x = self.fc_x(x).view(bs, -1)  # 3->1 [batch_size, 20] -> [batch_size, 20]
        y = self.fc_y(y)  # 线性层 [batch_size, 10] -> [batch_size, 20]
        # 合在一起
        z = torch.cat((x, y), dim=1) #[batch_size, 20] -> [batch_size, 40]
        out = self.fc(z) #[batch_size, 40] -> [batch_size, 2]

        return out


model = CNN()
# print(model)

