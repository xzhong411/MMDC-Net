""" Parts of the U-Net model """
from torchvision.transforms import ToPILImage, transforms

"""https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Conv(nn.Module):
    """(convolution => [BN] => ReLU) """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down_Cat(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
        )
        self.doubleconv = DoubleConv(in_channels+out_channels, out_channels)
    def forward(self, x1,x2):
        x1=self.maxpool_conv(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.doubleconv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x1):
        x = self.maxpool_conv(x1)
        return self.relu(x)

class NEWMODEL_111(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv3_1 =  nn.Conv2d(out_channels*2,out_channels*2, kernel_size=1)
        self.conv3_3_1 =  nn.Sequential(nn.Conv2d(out_channels*2,out_channels*2, kernel_size=3, padding=1, dilation=1),
                                        nn.ReLU(inplace=True))
        self.conv3_3_3 =  nn.Sequential(nn.Conv2d(out_channels*2,out_channels*2, kernel_size=3, padding=3, dilation=3),
                                        nn.ReLU(inplace=True))
        self.conv3_3_5 =  nn.Sequential(nn.Conv2d(out_channels*2,out_channels*2, kernel_size=3, padding=5, dilation=5),
                                        nn.ReLU(inplace=True))
        self.conv33 =  nn.Sequential(nn.Conv2d(out_channels*2,out_channels, kernel_size=1),
                                        nn.ReLU(inplace=True))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_1 = nn.Conv2d(out_channels // 2, out_channels //2, kernel_size=1)
        self.conv1_3_1 = nn.Sequential(
            nn.Conv2d(out_channels //2, out_channels //2, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True))
        self.conv1_3_3 = nn.Sequential(
            nn.Conv2d(out_channels //2, out_channels //2, kernel_size=3, padding=3, dilation=3),
            nn.ReLU(inplace=True))
        self.conv1_3_5 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=5, dilation=5),
            nn.ReLU(inplace=True))
        self.max = nn.Sequential(nn.MaxPool2d(2),
                                 nn.Conv2d(out_channels // 2, out_channels, kernel_size=1),
                                 nn.BatchNorm2d(out_channels))
    def forward(self, x1,x2,x3):
        """ x1--> 2H * 2W * C/2,  x2--> H * W * C,  x3-->H/2 * W/2 * 2C"""
        x1_1 = self.conv1_1(x1)
        x1_2 = self.conv1_1(self.conv1_3_1(x1))
        x1_3 = self.conv1_1(self.conv1_3_3(self.conv1_3_1(x1)))
        x1_4 = self.conv1_1(self.conv1_3_5(self.conv1_3_3(self.conv1_3_1(x1))))
        # print("x1_4{}".format(x1_4.shape))
        x11 = self.max(x1_1 + x1_2 + x1_3 + x1_4)
        # print("x11{}".format(x11.shape))
        x3_1 = self.conv3_1(x3)
        x3_2 = self.conv3_1(self.conv3_3_1(x3))
        x3_3 = self.conv3_1(self.conv3_3_3(self.conv3_3_1(x3)))
        x3_4 = self.conv3_1(self.conv3_3_5(self.conv3_3_3(self.conv3_3_1(x3))))
        x33 = self.conv33(self.up(x3_1 + x3_2 + x3_3 +x3_4))
        return x11 + x2 + x33
class NEWMODEL_110(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv1_1 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=1)
        self.conv1_3_1 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True))
        self.conv1_3_3 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=3, dilation=3),
            nn.ReLU(inplace=True))
        self.conv1_3_5 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=5, dilation=5),
            nn.ReLU(inplace=True))
        self.max = nn.Sequential(nn.MaxPool2d(2),
                                 nn.Conv2d(out_channels // 2, out_channels, kernel_size=1),
                                 nn.BatchNorm2d(out_channels))
    def forward(self, x1,x2):
        """ x1--> 2H * 2W * C/2,  x2--> H * W * C"""
        x1_1 = self.conv1_1(x1)
        x1_2 = self.conv1_1(self.conv1_3_1(x1))
        x1_3 = self.conv1_1(self.conv1_3_3(self.conv1_3_1(x1)))
        x1_4 = self.conv1_1(self.conv1_3_5(self.conv1_3_3(self.conv1_3_1(x1))))
        x11 = self.max(x1_1 + x1_2 + x1_3 + x1_4)
        return x11 + x2

class NEWMODEL_011(nn.Module):
    def __init__(self,  out_channels):
        super().__init__()
        self.conv3_1 =  nn.Conv2d(out_channels*2,out_channels *2, kernel_size=1)
        self.conv3_3_1 =  nn.Sequential(nn.Conv2d(out_channels*2,out_channels*2, kernel_size=3, padding=1, dilation=1),
                                        nn.ReLU(inplace=True))
        self.conv3_3_3 =  nn.Sequential(nn.Conv2d(out_channels*2,out_channels*2, kernel_size=3, padding=3, dilation=3),
                                        nn.ReLU(inplace=True))
        self.conv3_3_5 =  nn.Sequential(nn.Conv2d(out_channels*2,out_channels*2, kernel_size=3, padding=5, dilation=5),
                                        nn.ReLU(inplace=True))
        self.conv33 =  nn.Sequential(nn.Conv2d(out_channels*2,out_channels, kernel_size=1),
                                        nn.ReLU(inplace=True))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x2,x3):
        """x2--> H * W * C,  x3-->H/2 * W/2 * 2C"""
        x3_1 = self.conv3_1(x3)
        x3_2 = self.conv3_1(self.conv3_3_1(x3))
        x3_3 = self.conv3_1(self.conv3_3_3(self.conv3_3_1(x3)))
        x3_4 = self.conv3_1(self.conv3_3_5(self.conv3_3_3(self.conv3_3_1(x3))))
        x33 = self.conv33(self.up(x3_1 + x3_2 + x3_3 +x3_4))
        return  x2 + x33
#
# class Down_Cat(nn.Module):
#     """Downscaling with maxpool then double conv"""
#
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.MaxPool2d(2)
#         self.diaconv1 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1),
#                                       nn.ReLU(inplace=True))
#         self.diaconv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2),
#                                       nn.ReLU(inplace=True))
#         self.diaconv3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4),
#                                       nn.ReLU(inplace=True))
#         self.diaconv4 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=6, dilation=6),
#                                       nn.ReLU(inplace=True))
#         self.conv = Conv(in_channels*4+out_channels, out_channels)
#         self.bn = nn.BatchNorm2d(out_channels*4)
#         self.conv1_relu = nn.Sequential(nn.Conv2d(out_channels*4, out_channels*4, kernel_size=1, padding=0),
#                                       nn.ReLU(inplace=True))
#     def forward(self, x1,x2):
#         x1=self.maxpool_conv(x1)
#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         x1 = torch.cat([self.diaconv1(x), self.diaconv2(x), self.diaconv3(x), self.diaconv4(x)], dim=1)
#         x1 = self.bn(x1)
#         return self.conv1_relu(x1)



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16,stride=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1,x2,x3,x4,x5):
        x = torch.cat([x1, x2,x3,x4,x5], dim=1)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        b, c, _, _ = out.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        out += residual
        out = self.relu(out)
        return out

class OutLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(OutLayer, self).__init__()

        self.fc = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//5, kernel_size=1, padding=0),
            nn.BatchNorm2d(input_channels//5),
            nn.ReLU(inplace=False),
            nn.Conv2d(input_channels//5, input_channels // 5, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channels // 5),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels//5, output_channels, kernel_size=1, padding=0),
           )

    def forward(self, x):
        x = self.fc(x)
        outlayer = nn.Sigmoid()
        return outlayer(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)
        # self.contra=nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        # 这个上采用需要设置其输入通道，输出通道.其中kernel_size、stride
        # 大小要跟对应下采样设置的值一样大小。这样才可恢复到相同的wh。这里时反卷积操作。
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        # 这里不会改变通道数，其中scale_factor是上采用的放大因子，其是相对于当前的输入大小的倍数
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=(in_channels//32), align_corners=True))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Sequential(nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1),
        nn.BatchNorm2d(out_channels))
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.up = nn.Sequential(
            upconv2x2(in_channels, out_channels,mode=""),
            conv1x1(in_channels, out_channels),
        )
        # nn.Upsample(scale_factor=in_channels//32, mode='nearest'),
        # nn.Conv2d(out_channels,out_channels, kernel_size=1)

    def forward(self, x):
        x=self.up(x)
        outlayer = nn.Sigmoid()
        return outlayer(x)


class OutConv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels, kernel_size=1),
        )

    def forward(self, x):
        x=self.conv(x)
        outlayer = nn.Sigmoid()
        return outlayer(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class RecallCrossEntropy(nn.Module):
    def __init__(self, n_classes=2, ignore_index=255):
        super(RecallCrossEntropy, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # input (batch,n_classes,H,W)
        # target (batch,H,W)
        pred = input.argmax(1)

        idex = (pred != target).view(-1)

        # calculate ground truth counts
        gt_counter = torch.ones((self.n_classes,)).cuda()
        gt_idx, gt_count = torch.unique(target, return_counts=True)

        # map ignored label to an exisiting one
        gt_count[gt_idx == self.ignore_index] = gt_count[1].clone()
        gt_idx[gt_idx == self.ignore_index] = 1
        gt_counter[gt_idx] = gt_count.float()

        # calculate false negative counts
        fn_counter = torch.ones((self.n_classes)).cuda()
        fn = target.view(-1)[idex]
        fn_idx, fn_count = torch.unique(fn, return_counts=True)

        # map ignored label to an exisiting one
        fn_count[fn_idx == self.ignore_index] = fn_count[0].clone()

        fn_idx[fn_idx == self.ignore_index] = 1
        fn_counter[fn_idx] = fn_count.float()

        weight = fn_counter / gt_counter

        CE = F.cross_entropy(input, target, reduction='none', ignore_index=self.ignore_index)
        loss = weight[target] * CE
        return loss.mean()