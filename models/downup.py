import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DownUp(nn.Module):
    """
    """

    def __init__(self, in_channels=1, out_channels=2,
                 final_sigmoid=False,
                 max_pool_kernel_size=2,
                 conv_layer_order='cbr',
                 init_channel_number=64):
        super(DownUp, self).__init__()

        # encoder path consist of 4 subsequent Encoder modules
        # the number of features maps is the same as in the paper
        self.encoders = nn.ModuleList([
            Encoder(in_channels, init_channel_number, conv_kernel_size=3, is_max_pool=False,
                    max_pool_strides=1, conv_layer_order=conv_layer_order),
            Encoder(init_channel_number, 2 * init_channel_number, conv_kernel_size=3, is_max_pool=True,
                    max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=2,
                    conv_layer_order=conv_layer_order),
            Encoder(2 * init_channel_number, 4 * init_channel_number, conv_kernel_size=3,
                    is_max_pool=True,
                    max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=2,
                    conv_layer_order=conv_layer_order),
            Encoder(4 * init_channel_number, 8 * init_channel_number, conv_kernel_size=3,
                    is_max_pool=True,
                    max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=2,
                    conv_layer_order=conv_layer_order)
        ])

        self.decoders = nn.ModuleList([
            Decoder(8 * init_channel_number, 4 * init_channel_number, trans_kernel_size=3, scale_factor=2,
                    conv_layer_order=conv_layer_order),
            Decoder(4 * init_channel_number, 2 * init_channel_number, trans_kernel_size=3, scale_factor=2,
                    conv_layer_order=conv_layer_order),
            Decoder(2 * init_channel_number, init_channel_number, trans_kernel_size=3, scale_factor=2,
                    conv_layer_order=conv_layer_order)
        ])

        # in the last layer a 1×1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(init_channel_number, out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # encoder part
        shape = x.shape
        sizes = []
        for encoder in self.encoders:
            x = encoder(x)
            sizes.insert(0, x.shape)
        sizes = sizes[1:]

        # decoder part
        for decoder, size in zip(self.decoders, sizes):
            x = decoder(size, x)

        x = self.final_conv(x)
        if not self.training:
            x = self.final_activation(x)
        X_diff = shape[2] - x.shape[2]
        Y_diff = shape[3] - x.shape[3]
        Z_diff = shape[4] - x.shape[4]
        if X_diff + Y_diff + Z_diff != 0:
            x = F.pad(x, (0, Z_diff, 0, Y_diff, 0, X_diff))
        return x


class DoubleConv(nn.Sequential):
    """
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='crb'):
        super(DoubleConv, self).__init__()
        if in_channels < out_channels:
            # if in_channels < out_channels we're in the encoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels // 2
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # otherwise we're in the decoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels
        # conv1
        self._add_conv(1, conv1_in_channels, conv1_out_channels, kernel_size, order)
        # conv2
        self._add_conv(2, conv2_in_channels, conv2_out_channels, kernel_size, order)

    def _add_conv(self, pos, in_channels, out_channels, kernel_size, order):
        """
        """
        assert pos in [1, 2, 3], 'pos MUST be either 1 or 2'
        assert 'c' in order, "'c' (conv layer) MUST be present"
        assert 'r'or's' in order, "'r' (ReLU layer) MUST be present"
        assert order[0] is not 'r', 'ReLU cannot be the first operation in the layer'

        for i, char in enumerate(order):
            if char == 'r':
                self.add_module('relu{}'.format(pos), nn.LeakyReLU(inplace=True))
            elif char == 's':
                self.add_module('sigmoid{}'.format(pos), nn.Sigmoid())
            elif char == 'c':
                self.add_module('conv{}'.format(pos), nn.Conv3d(in_channels,
                                                                out_channels,
                                                                kernel_size,
                                                                padding=((kernel_size - 1) // 2)))
            elif char == 'g':
                is_before_conv = i < order.index('c')
                assert not is_before_conv, 'GroupNorm3d MUST go after the Conv3d'
                # self.add_module('norm{}'.format(pos), groupnorm.GroupNorm3d(out_channels))
                self.add_module('norm{}'.format(pos), nn.GroupNorm(1, out_channels))
            elif char == 'b':
                is_before_conv = i < order.index('c')
                if is_before_conv:
                    self.add_module('norm{}'.format(pos), nn.BatchNorm3d(in_channels))
                else:
                    self.add_module('norm{}'.format(pos), nn.BatchNorm3d(out_channels))
            else:
                raise ValueError(
                    "Unsupported layer type '{}'. MUST be one of 'b', 'r', 'c'".format(char))


class Encoder(nn.Module):
    """
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, is_max_pool=True,
                 max_pool_kernel_size=3, max_pool_strides=1, conv_layer_order='crg'):
        super(Encoder, self).__init__()
        self.max_pool_size = max_pool_kernel_size
        self.max_pool = nn.MaxPool3d(kernel_size=max_pool_kernel_size, stride=max_pool_strides,
                                     padding=(max_pool_kernel_size - 1) // 2) if is_max_pool else None
        self.double_conv = DoubleConv(in_channels, out_channels,
                                      kernel_size=conv_kernel_size,
                                      order=conv_layer_order)

    def forward(self, x):
        if self.max_pool is not None:
            # X_diff = x.shape[2] % self.max_pool_size
            # Y_diff = x.shape[3] % self.max_pool_size
            # Z_diff = x.shape[4] % self.max_pool_size
            # if X_diff + Y_diff + Z_diff != 0:
            #     x = F.pad(x, (0, Z_diff, 0, Y_diff, 0, X_diff))
            x = self.max_pool(x)
        x = self.double_conv(x)
        return x


class Decoder(nn.Module):
    """
    """

    def __init__(self, in_channels, out_channels, trans_kernel_size=3,
                 scale_factor=1, conv_layer_order='crg'):
        super(Decoder, self).__init__()
        self.scale_factor = scale_factor
        self.double_conv = DoubleConv(in_channels, out_channels,
                                      kernel_size=trans_kernel_size,
                                      order=conv_layer_order)

    def forward(self, size, x):
        x = F.interpolate(x, scale_factor=2)
        # X_diff = size[2] - x.shape[2]
        # Y_diff = size[3] - x.shape[3]
        # Z_diff = size[4] - x.shape[4]
        # if X_diff + Y_diff + Z_diff != 0:
        #     x = F.pad(x, (0, Z_diff, 0, Y_diff, 0, X_diff))
        # concatenate encoder_features (encoder path) with the upsampled input across channel dimension
        x = self.double_conv(x)
        return x

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    data = torch.zeros((1, 1, 96, 96, 370)).to(device)
    model = DownUp(out_channels=2, init_channel_number=4).to(device)
    # print(model)
    output = model(data)
    # print(output.shape)


if __name__ == '__main__':
    main()