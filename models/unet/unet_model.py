""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.out = []
        self.out.append(res(128, 1).cuda())
        self.out.append(res(256, 1).cuda())
        self.out.append(res(512, 1).cuda())
        self.out.append(res(512, 1).cuda())

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.fout = []
        self.fout.append(after_res(64, 64).cuda())
        self.fout.append(after_res(128, 64).cuda())
        self.fout.append(after_res(256, 64).cuda())
        self.fout.append(after_res(512, 64).cuda())

        self.gate = after_res(64, 64).cuda()

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        down_x = [x2, x3, x4, x5]
        put_x = []
        for i in range(4):
            put_x.append(self.out[i](down_x[i]))

        after_x = []
        after_x.append(x5)
        x = self.up1(x5, x4)
        after_x.append(x)
        x = self.up2(x, x3)
        after_x.append(x)
        x = self.up3(x, x2)
        after_x.append(x)
        x = self.up4(x, x1)

        gs = []
        xs = []
        for i in range(len(after_x)):
            xx, g = self.fout[i](after_x[3 - i])
            gs.append(g)
            xs.append(xx)
        other = gs[0] * xs[0]
        for i in range(1, len(gs)):
            other += gs[i] * xs[i]

        x, gx = self.gate(x)
        x = other * (1 - gx) + gx * x + x

        logits = self.outc(x)
        y = F.sigmoid(logits)

        return put_x, y
