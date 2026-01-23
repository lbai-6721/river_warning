import torch
import torch.nn as nn

from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2

class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x



class UpConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self, ch_in=3, ch_out=2, backbone="mobilenet", downsample_factor=16,pretrained=True):
        super().__init__()
        if backbone=="xception":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            #----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="mobilenet":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            #----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))
        feature_channels = [8, 16, 32, 64, 128]

        # 定义四个池化层，池化核尺寸均为2，步长均为2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 根据通道设定，定义下采样过程中使用的五个卷积层
        self.conv1 = ConvBlock(ch_in, feature_channels[0])
        self.conv2 = ConvBlock(feature_channels[0], feature_channels[1])
        self.conv3 = ConvBlock(feature_channels[1], feature_channels[2])
        self.conv4 = ConvBlock(feature_channels[2], feature_channels[3])
        self.conv5 = ConvBlock(feature_channels[3], feature_channels[4])

        # 根据通道设定，定义上次样过程中使用的卷积层和上采样层
        self.up5 = UpConvBlock(feature_channels[4], feature_channels[3])
        self.up_conv5 = ConvBlock(feature_channels[4], feature_channels[3])
        self.up4 = UpConvBlock(feature_channels[3], feature_channels[2])
        self.up_conv4 = ConvBlock(feature_channels[3], feature_channels[2])
        self.up3 = UpConvBlock(feature_channels[2], feature_channels[1])
        self.up_conv3 = ConvBlock(feature_channels[2], feature_channels[1])
        self.up2 = UpConvBlock(feature_channels[1], feature_channels[0])
        self.up_conv2 = ConvBlock(feature_channels[1], feature_channels[0])

        # 定义最后一层卷积，输出通道数等于目标类别数
        self.conv_last = nn.Conv2d(feature_channels[0], ch_out, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 第一层卷积，将3通道的输入变换为特征图
        f1 = self.conv1(x)

        # 计算下采样过程，通过池化和卷积获得中间特征图
        f2 = self.pool1(f1)
        f2 = self.conv2(f2)
        f3 = self.pool2(f2)
        f3 = self.conv3(f3)
        f4 = self.pool3(f3)
        f4 = self.conv4(f4)
        f5 = self.pool4(f4)
        f5 = self.conv5(f5)

        # 第一次特征融合
        up_f5 = self.up5(f5)
        up_f5 = torch.cat((f4, up_f5), dim=1)
        up_f5 = self.up_conv5(up_f5)

        # 第二次特征融合
        up_f4 = self.up4(up_f5)
        up_f4 = torch.cat((f3, up_f4), dim=1)
        up_f4 = self.up_conv4(up_f4)

        # 第三次特征融合
        up_f3 = self.up3(up_f4)
        up_f3 = torch.cat((f2, up_f3), dim=1)
        up_f3 = self.up_conv3(up_f3)

        # 第四次特征融合
        up_f2 = self.up2(up_f3)
        up_f2 = torch.cat((f1, up_f2), dim=1)
        up_f2 = self.up_conv2(up_f2)

        # 计算最后一层卷积输出，获得预测mask
        mask = self.conv_last(up_f2)
        mask = self.sigmoid(mask)
        return mask


if __name__ == "__main__":
    unet = UNet()
    x = torch.randn(size=(1, 3, 256, 256))
    print(unet(x).size())
    # torch.onnx.export(model=unet, args=x,
    #                   f="unet.onnx", input_names=["input"],
    #                   output_names=["output"], opset_version=11)
