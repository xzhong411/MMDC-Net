""" The MMDC-Net model """


import torch
import torch.nn as nn
import torch.nn.functional as F



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

class MMDC_111(nn.Module):
    """MMDC module"""
    def __init__(self, out_channels):
        super().__init__()
        self.conv3_1 =  nn.Conv2d(out_channels*2,out_channels*2, kernel_size=1)
        self.conv3_3_1 =  nn.Sequential(nn.Conv2d(out_channels*2,out_channels*2, kernel_size=3, padding=1, dilation=1),
                                        nn.ReLU(inplace=True))                   ## conv3_3_x indicates denotes the convolution of different dilation at the lower level
        self.conv3_3_3 =  nn.Sequential(nn.Conv2d(out_channels*2,out_channels*2, kernel_size=3, padding=3, dilation=3),
                                        nn.ReLU(inplace=True))
        self.conv3_3_5 =  nn.Sequential(nn.Conv2d(out_channels*2,out_channels*2, kernel_size=3, padding=5, dilation=5),
                                        nn.ReLU(inplace=True))
        self.conv33 =  nn.Sequential(nn.Conv2d(out_channels*2,out_channels, kernel_size=1),
                                        nn.ReLU(inplace=True))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_1 = nn.Conv2d(out_channels // 2, out_channels //2, kernel_size=1)         # conv1_3_x indicates denotes the convolution of different dilation at the upper level
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


class MMDC_011(nn.Module):
    """MMDC module"""
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

class SELayer(nn.Module):
    """[X1,X2,X3,X4,X5]-->SE"""
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
    """Out:  n_channels from 320 to 2"""
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


class RecallCrossEntropy(nn.Module):
    """recall loss"""
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
