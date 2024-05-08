# -*- coding: utf-8 -*-
# @Time    : 2024/5/6 下午12:24
# @Author  : yang chen
import torch
import torch.nn as nn
class ConvBNReLUx2(nn.Module):
    '''
    卷积模块,包含CONV-BN-RELU-CONV-BN-RELU.
    '''

    def __init__(self, in_channels, out_channels):
        super(ConvBNReLUx2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpSample(nn.Module):
    '''
    上采样模块,包含UP2x-CONV-BN-RELU
    '''

    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()

        base_channel = 64
        channels = [base_channel, base_channel * 2, base_channel * 4, base_channel * 8, base_channel * 16]

        self.Maxpool2 = nn.MaxPool2d(2, 2)
        self.Maxpool3 = nn.MaxPool2d(2, 2)
        self.Maxpool4 = nn.MaxPool2d(2, 2)
        self.Maxpool5 = nn.MaxPool2d(2, 2)

        self.Conv1 = ConvBNReLUx2(in_channels, channels[0])  # stem
        self.Conv2 = ConvBNReLUx2(channels[0], channels[1])  #
        self.Conv3 = ConvBNReLUx2(channels[1], channels[2])  #
        self.Conv4 = ConvBNReLUx2(channels[2], channels[3])  #
        self.Conv5 = ConvBNReLUx2(channels[3], channels[4])  #

        self.Up5 = UpSample(channels[4], channels[3])
        self.Up5_Conv = ConvBNReLUx2(channels[4], channels[3])

        self.Up4 = UpSample(channels[3], channels[2])
        self.Up4_Conv = ConvBNReLUx2(channels[3], channels[2])

        self.Up3 = UpSample(channels[2], channels[1])
        self.Up3_Conv = ConvBNReLUx2(channels[2], channels[1])

        self.Up2 = UpSample(channels[1], channels[0])
        self.Up2_Conv = ConvBNReLUx2(channels[1], channels[0])

        self.out = nn.Conv2d(channels[0],num_classes,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        # stem
        e1 = self.Conv1(x)  # 256/3->256/64

        #
        e2 = self.Maxpool2(e1)  #
        e2 = self.Conv2(e2)  # 256/64->128/128
        #
        e3 = self.Maxpool3(e2)  #
        e3 = self.Conv3(e3)  # 128/128 -> 64/256
        #
        e4 = self.Maxpool4(e3)
        e4 = self.Conv4(e4) #64/256->32/512
        #
        e5 = self.Maxpool5(e4)
        e5 = self.Conv5(e5) #32/512->16/1024

        #
        d5 = self.Up5(e5)
        d5 = torch.cat((e4,d5),dim=1)
        d5 = self.Up5_Conv(d5)
        #
        d4 = self.Up4(d5)
        d4 = torch.cat((e3,d4),dim=1)
        d4 = self.Up4_Conv(d4)
        #
        d3 = self.Up3(d4)
        d3 = torch.cat((e2,d3),dim=1)
        d3 = self.Up3_Conv(d3)
        #
        d2 = self.Up2(d3)
        d2 = torch.cat((e1,d2),dim=1)
        d2 = self.Up2_Conv(d2)

        out = self.out(d2)
        return out