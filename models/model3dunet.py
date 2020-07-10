import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UNet3D(nn.Module):
    """
    """

    def __init__(self, in_channels, out_channels,
                 final_sigmoid,
                 conv_kernel_size=3,
                 trans_kernel_size=3,
                 scale_factor=2,
                 max_pool_kernel_size=2,
                 max_pool_strides=2,
                 interpolate=True,
                 conv_layer_order='cbr',
                 init_channel_number=64):
        super(UNet3D, self).__init__()

        # encoder path consist of 4 subsequent Encoder modules
        # the number of features maps is the same as in the paper
        self.encoders = nn.ModuleList([
            Encoder(in_channels, init_channel_number, conv_kernel_size=conv_kernel_size, is_max_pool=False,
                    max_pool_strides=1, conv_layer_order=conv_layer_order),
            Encoder(init_channel_number, 2 * init_channel_number, conv_kernel_size=conv_kernel_size, is_max_pool=True,
                    max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=max_pool_strides,
                    conv_layer_order=conv_layer_order),
            Encoder(2 * init_channel_number, 4 * init_channel_number, conv_kernel_size=conv_kernel_size,
                    is_max_pool=True,
                    max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=max_pool_strides,
                    conv_layer_order=conv_layer_order),
            Encoder(4 * init_channel_number, 8 * init_channel_number, conv_kernel_size=conv_kernel_size,
                    is_max_pool=True,
                    max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=max_pool_strides,
                    conv_layer_order=conv_layer_order)
        ])

        self.decoders = nn.ModuleList([
            Decoder(4 * init_channel_number + 8 * init_channel_number, 4 * init_channel_number, interpolate,
                    trans_kernel_size=trans_kernel_size, scale_factor=scale_factor,
                    conv_layer_order=conv_layer_order),
            Decoder(2 * init_channel_number + 4 * init_channel_number, 2 * init_channel_number, interpolate,
                    trans_kernel_size=trans_kernel_size, scale_factor=scale_factor,
                    conv_layer_order=conv_layer_order),
            Decoder(init_channel_number + 2 * init_channel_number, init_channel_number, interpolate,
                    trans_kernel_size=trans_kernel_size, scale_factor=scale_factor,
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
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        x = self.final_conv(x)
        if not self.training:
            x = self.final_activation(x)
        return x


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d)
    with the number of output channels 'out_channels // 2' and 'out_channels' respectively.
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be change however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ReLU use order='cbr'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cbr'):
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
        """Add the conv layer with non-linearity and optional batchnorm

        Args:
            pos (int): the order (position) of the layer. MUST be 1 or 2
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            order (string): order of things, e.g.
                'cr' -> conv + ReLU
                'crg' -> conv + ReLU + groupnorm
            num_groups (int): number of groups for the GroupNorm
        """
        assert pos in [1, 2, 3], 'pos MUST be either 1 or 2'
        assert 'c' in order, "'c' (conv layer) MUST be present"
        assert 'r'or's'or'l' in order, "'r' (ReLU layer) MUST be present"
        assert order[0] is not 'r'or'l', 'ReLU cannot be the first operation in the layer'

        for i, char in enumerate(order):
            if char == 'r':
                self.add_module('relu{}'.format(pos), nn.ReLU(inplace=True))
            elif char == 'l':
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
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int): size of the convolving kernel
        is_max_pool (bool): if True use MaxPool3d before DoubleConv
        max_pool_kernel_size (tuple): the size of the window to take a max over
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
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
            X_diff = x.shape[2] % self.max_pool_size
            Y_diff = x.shape[3] % self.max_pool_size
            Z_diff = x.shape[4] % self.max_pool_size
            if X_diff + Y_diff + Z_diff != 0:
                x = F.pad(x, (0, Z_diff, 0, Y_diff, 0, X_diff))
            x = self.max_pool(x)
        x = self.double_conv(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose3d or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        interpolate (bool): if True use nn.Upsample for upsampling, otherwise
            learn ConvTranspose3d if you have enough GPU memory and ain't
            afraid of overfitting
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, interpolate, trans_kernel_size=3,
                 scale_factor=1, conv_layer_order='crg'):
        super(Decoder, self).__init__()
        self.scale_factor = scale_factor
        if interpolate:
            self.upsample = None
        else:
            # make sure that the output size reverses the MaxPool3d
            # D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0]
            self.upsample = nn.ConvTranspose3d(2 * out_channels,
                                               2 * out_channels,
                                               kernel_size=trans_kernel_size,
                                               stride=scale_factor,
                                               padding=((trans_kernel_size - 1) // 2),
                                               output_padding=scale_factor - 1)
        self.double_conv = DoubleConv(in_channels, out_channels,
                                      kernel_size=trans_kernel_size,
                                      order=conv_layer_order)

    def forward(self, encoder_features, x):
        if self.upsample is None:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            x = self.upsample(x)
        # print(x.shape, encoder_features.shape)
        X_diff = encoder_features.shape[2] - x.shape[2]
        Y_diff = encoder_features.shape[3] - x.shape[3]
        Z_diff = encoder_features.shape[4] - x.shape[4]
        if X_diff + Y_diff + Z_diff != 0:
            x = F.pad(x, (0, Z_diff, 0, Y_diff, 0, X_diff))
        # concatenate encoder_features (encoder path) with the upsampled input across channel dimension
        x = torch.cat((encoder_features, x), dim=1)
        x = self.double_conv(x)
        return x

