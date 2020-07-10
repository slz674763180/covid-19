import importlib
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self,  in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_relu = with_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_relu:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)




class ResBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 1024, layers[4], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_f = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return [x_f, x0, x1, x2, x3, x4, x5]


class res(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x=self.bridge(x)
        x = F.interpolate(x, (512, 512), mode='bilinear')
        return x


class ResUnet(nn.Module):
    def __init__(self, n_classes=1, **kwargs):
        super().__init__()
        self.ResNet = ResNet(ResBlock, [2, 2, 2, 2, 2])
        self.bridge = Bridge(1024, 1024)
        up_blocks = []
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(256, 128))
        up_blocks.append(UpBlockForUNetWithResNet50(128, 64))
        up_blocks.append(UpBlockForUNetWithResNet50(64, 32))
        up_blocks.append(UpBlockForUNetWithResNet50(17, 16, 32, 16))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = res(16)
        self.out1 = res(32)
        self.out2 = res(64)
        self.out3 = res(128)
        self.out4 = res(256)
        self.out5 = res(512)
        self.out6 = res(1024)

    def forward(self, x, with_output_feature_map=False):

        down_x = self.ResNet(x)
        x = self.bridge(down_x[6])

        put_1 = self.out1(down_x[1])
        put_2 = self.out2(down_x[2])
        put_3 = self.out3(down_x[3])
        put_4 = self.out4(down_x[4])
        put_5 = self.out5(down_x[5])
        put_6 = self.out6(down_x[6])
        put_after = []
        put_x = [put_1,put_2,put_3,put_4,put_5,put_6]

        for i, block in enumerate(self.up_blocks, 1):
            put_after.append(x)
            x = block(x, down_x[6 - i])
        x = self.out(x)
        # put_after.append(x)
        #
        # put_after[0] = self.out6(put_after[0])
        # put_after[1] = self.out5(put_after[1])
        # put_after[2] = self.out4(put_after[2])
        # put_after[3] = self.out3(put_after[3])
        # put_after[4] = self.out2(put_after[4])
        # put_after[5] = self.out1(put_after[5])
        # put_after[6] = self.out(put_after[6])
        return put_x, x
