import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


########################
# Convolutional Blocks #
########################

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
                 transpose=False, output_padding=0):
        super().__init__()
        if padding == -1:
            padding = ((np.array(kernel_size) - 1) * np.array(dilation)) // 2
            # to check if padding is not a 0-d array, otherwise tuple(padding) will raise an exception
            if hasattr(padding, '__iter__'):
                padding = tuple(padding)

        if transpose:
            self.conv = nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size,
                stride, padding, output_padding, groups, bias, dilation)
        else:
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                stride, padding, dilation, groups, bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm1d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm1d(out_channels, track_running_stats=True)
        elif norm == "SN":
            self.norm = None
            self.conv = nn.utils.spectral_norm(self.conv)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f"Norm type {norm} not implemented")

        self.activation = activation

    def forward(self, xs):
        out = self.conv(xs)
        if self.activation is not None:
            out = self.activation(out)
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class VanillaConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True), conv_dim='1d'
    ):

        super().__init__()
        if conv_dim == '1d':
            self.module = torch.nn
        else:
            raise NotImplementedError(f'conv_dim {conv_dim} is not implemented.')

        self.padding = ((kernel_size - 1) * dilation) // 2 if padding == -1 else padding
        self.featureConv = self.module.Conv1d(
            in_channels, out_channels, kernel_size,
            stride, self.padding, dilation, groups, bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = self.module.BatchNorm1d(out_channels)
        elif norm == "IN":
            self.norm_layer = self.module.InstanceNorm1d(out_channels, track_running_stats=True)
        elif norm == "SN":
            self.norm = None
            self.featureConv = nn.utils.spectral_norm(self.featureConv)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f"Norm type {norm} not implemented")

        self.activation = activation

    def forward(self, xs):
        out = self.featureConv(xs)
        if self.activation:
            out = self.activation(out)
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class VanillaDeconv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
        scale_factor=2, conv_dim='1d'
    ):
        super().__init__()
        self.conv = VanillaConv(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation, conv_dim=conv_dim)
        self.scale_factor = scale_factor

    def forward(self, xs):
        xs_resized = F.interpolate(xs, scale_factor=self.scale_factor)
        return self.conv(xs_resized)
