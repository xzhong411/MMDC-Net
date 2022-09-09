import torch
import torch.nn as nn
import torch.nn.functional as F

from core.unet_parts import *



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.pool = nn.MaxPool2d((2, 2))
        self.conv = DoubleConv(n_channels, 32)

        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.newmodel1 = NEWMODEL_011(32)
        self.newmodel2 = NEWMODEL_111(64)
        self.newmodel3 = NEWMODEL_111(128)
        self.newmodel4 = NEWMODEL_111(256)
        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.out = OutConv1(32, n_classes)
        self.out6 = OutConv(256, n_classes)
        self.out7 = OutConv(128, n_classes)
        self.out8 = OutConv(64, n_classes)
        self.out9 = OutConv1(32, n_classes)
        self.side_5 = nn.Conv2d(256, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_9 = nn.Conv2d(32, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.out_layer = OutLayer(320, 2)
        self.se = SELayer(320)
    def forward(self, x):
        _, _, img_shape, _ = x.size()

        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x1_1 = self.newmodel1(x1,x2)
        x2_1 = self.newmodel2(x1,x2,x3)
        x3_1 = self.newmodel3(x2,x3,x4)
        x4_1 = self.newmodel4(x3,x4,x5)
        x6 = self.up1(x5, x4_1)
        x7 = self.up2(x6, x3_1)
        x8 = self.up3(x7, x2_1)
        x9 = self.up4(x8, x1_1)

        side_5 = F.interpolate(x6, size=(img_shape, img_shape), mode='bilinear', align_corners=True)
        side_6 = F.interpolate(x7, size=(img_shape, img_shape), mode='bilinear', align_corners=True)
        side_7 = F.interpolate(x8, size=(img_shape, img_shape), mode='bilinear', align_corners=True)
        side_8 = F.interpolate(x9, size=(img_shape, img_shape), mode='bilinear', align_corners=True)
        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)
        side_9 = self.side_9(x1)
        out = self.se(side_5, side_6, side_7, side_8,side_9)
        out = self.out_layer(out)

        return out
