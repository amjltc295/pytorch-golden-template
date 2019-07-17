import torch
import torch.nn as nn

from model.blocks import (VanillaConv, VanillaDeconv)


###########################
# Encoder/Decoder Modules #
###########################

class BaseModule(nn.Module):
    def __init__(self, conv_type):
        super().__init__()
        self.conv_type = conv_type

        if conv_type == 'vanilla':
            self.ConvBlock = VanillaConv
            self.DeconvBlock = VanillaDeconv


class DownSampleModule(BaseModule):
    def __init__(self, nc_in, nf, use_bias, norm, conv_dim, conv_type):
        super().__init__(conv_type)
        import global_variables
        if global_variables.global_config.get('skip_conv1_TSM', False):
            first_conv_dim = '2d' if conv_dim == '2dtsm' else conv_dim
        else:
            first_conv_dim = conv_dim

        self.conv1 = self.ConvBlock(
            nc_in, nf * 1, kernel_size=32, stride=1,
            padding=-1, bias=use_bias, norm=norm, conv_dim=first_conv_dim)

        # Downsample 1
        self.conv2 = self.ConvBlock(
            nf * 1, nf * 2, kernel_size=24, stride=2,
            padding=-1, bias=use_bias, norm=norm, conv_dim=conv_dim)
        self.conv3 = self.ConvBlock(
            nf * 2, nf * 2, kernel_size=20, stride=1,
            padding=10, bias=use_bias, norm=norm, conv_dim=conv_dim)
        # Downsample 2
        self.conv4 = self.ConvBlock(
            nf * 2, nf * 4, kernel_size=24, stride=2,
            padding=-1, bias=use_bias, norm=norm, conv_dim=conv_dim)
        self.conv5 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=20, stride=1,
            padding=-1, bias=use_bias, norm=norm, conv_dim=conv_dim)
        self.conv6 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=20, stride=1,
            padding=-1, bias=use_bias, norm=norm, conv_dim=conv_dim)

        # Dilated Convolutions
        self.dilated_conv1 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=20, stride=1,
            padding=-1, bias=use_bias, norm=norm, conv_dim=conv_dim, dilation=2)
        self.dilated_conv2 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=20, stride=1,
            padding=-1, bias=use_bias, norm=norm, conv_dim=conv_dim, dilation=4)
        self.dilated_conv3 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=20, stride=1,
            padding=-1, bias=use_bias, norm=norm, conv_dim=conv_dim, dilation=8)
        self.dilated_conv4 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=20, stride=1,
            padding=-1, bias=use_bias, norm=norm, conv_dim=conv_dim, dilation=16)
        self.conv7 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=20, stride=1, padding=10,
            bias=use_bias, norm=norm, conv_dim=conv_dim)
        self.conv8 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=20, stride=1, padding=10,
            bias=use_bias, norm=norm, conv_dim=conv_dim)

    def forward(self, inp):
        c1 = self.conv1(inp)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)

        a1 = self.dilated_conv1(c6)
        a2 = self.dilated_conv2(a1)
        a3 = self.dilated_conv3(a2)
        a4 = self.dilated_conv4(a3)

        c7 = self.conv7(a4)
        c8 = self.conv8(c7)
        return c8, c4, c2  # For skip connection


class UpSampleModule(BaseModule):
    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_dim, conv_type):
        super().__init__(conv_type)
        # Upsample 1
        self.deconv1 = self.DeconvBlock(
            nc_in, nf * 2, kernel_size=20, stride=1, padding=-1,
            bias=use_bias, norm=norm, conv_dim=conv_dim)
        self.conv9 = self.ConvBlock(
            nf * 2, nf * 2, kernel_size=20, stride=1, padding=10,
            bias=use_bias, norm=norm, conv_dim=conv_dim)
        # Upsample 2
        self.deconv2 = self.DeconvBlock(
            nf * 2, nf * 1, kernel_size=20, stride=1, padding=-1,
            bias=use_bias, norm=norm, conv_dim=conv_dim)
        self.conv10 = self.ConvBlock(
            nf * 1, nf // 2, kernel_size=20, stride=1, padding=10,
            bias=use_bias, norm=norm, conv_dim=conv_dim)
        self.conv11 = self.ConvBlock(
            nf // 2, nc_out, kernel_size=19, stride=1,
            padding=-1, bias=use_bias, norm=None, activation=None, conv_dim=conv_dim)

    def concat_feature(self, ca, cb):
        if self.conv_type == 'partial':
            ca_feature, ca_mask = ca
            cb_feature, cb_mask = cb
            feature_cat = torch.cat((ca_feature, cb_feature), 1)
            # leave only the later mask
            return feature_cat, ca_mask
        else:
            return torch.cat((ca, cb), 1)

    def forward(self, inp):
        c8, c4, c2 = inp
        d1 = self.deconv1(c8)
        c9 = self.conv9(d1)

        d2 = self.deconv2(c9)
        c10 = self.conv10(d2)
        c11 = self.conv11(c10)
        return c11
