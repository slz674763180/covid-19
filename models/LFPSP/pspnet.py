import torch
from torch import nn
from torch.nn import functional as F
from models.LFPSP import extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class res(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.bridge(x)
        x = F.interpolate(x, (512, 512), mode='bilinear')
        return x


class after_res(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bridge(x)
        x = F.interpolate(x, (512, 512), mode='bilinear')
        forg = self.conv(x)
        g = self.sigmoid(forg)
        return x, g


class FPSPNet(nn.Module):
    def __init__(self, n_classes=1, sizes=(1, 2, 3, 6), psp_size=2048, backend='resnet50',
                 pretrained=False):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.Sigmoid()
        )
        self.out = []
        self.out.append(res(64, 1).cuda())
        self.out.append(res(64, 1).cuda())
        self.out.append(res(256, 1).cuda())
        self.out.append(res(512, 1).cuda())
        self.out.append(res(1024, 1).cuda())
        self.out.append(res(2048, 1).cuda())

        self.fout = []
        self.fout.append(after_res(64, 64).cuda())
        self.fout.append(after_res(64, 64).cuda())
        self.fout.append(after_res(256, 64).cuda())
        self.fout.append(after_res(1024, 64).cuda())

    def forward(self, x):
        output = []

        fs = self.feats(x)
        f=fs[5]
        put_x = []
        for i in range(6):
            put_x.append(self.out[i](fs[i]))

        p = self.psp(f)
        p = self.drop_1(p)
        output.append(p)

        p = self.up_1(p)
        p = self.drop_2(p)
        output.append(p)

        p = self.up_2(p)
        p = self.drop_2(p)
        output.append(p)

        p = self.up_3(p)
        p = self.drop_2(p)
        output.append(p)

        gs = []
        xs = []
        for i in range(len(output)):
            xx, g = self.fout[i](output[3 - i])
            gs.append(g)
            xs.append(xx)
        other = 0
        for i in range(1, len(gs)):
            other += gs[i] * xs[i]

        x, gx = xs[0],gs[0]
        x = other * (1 - gx) + gx * x + x
        return self.final(x)
        # return put_x, self.final(p), self.final(x)

#
# model = PSPNet(1)
# input = torch.rand((1, 1, 256, 256))
# out = model(input)
# print(1)
