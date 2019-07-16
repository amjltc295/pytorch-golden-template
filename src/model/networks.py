import torch
import torch.nn as nn

from model.modules import UpSampleModule, DownSampleModule


##################
# Generators #
##################

class CoarseNet(nn.Module):

    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_dim, conv_type):
        super().__init__()
        self.conv_type = conv_type
        self.downsample_module = DownSampleModule(
            nc_in, nf, use_bias, norm, conv_dim, conv_type)
        self.upsample_module = UpSampleModule(
            nf * 4, nc_out, nf, use_bias, norm, conv_dim, conv_type)

    def preprocess(self, masked_imgs, masks, guidances):
        # B, L, C, H, W = masked.shape
        masked_imgs = masked_imgs.transpose(1, 2)
        masks = masks.transpose(1, 2)
        if self.conv_type == 'partial':
            if guidances is not None:
                raise NotImplementedError('Partial convolution does not support guidance')
            # the input and output of partial convolution are both tuple (imgs, mask)
            inp = (masked_imgs, masks)
        elif self.conv_type == 'gated' or self.conv_type == 'vanilla':
            guidances = torch.full_like(masks, 0.) if guidances is None else guidances.transpose(1, 2)
            inp = torch.cat([masked_imgs, masks, guidances], dim=1)
        else:
            raise NotImplementedError(f"{self.conv_type} not implemented")

        return inp

    def postprocess(self, masked_imgs, masks, c11):
        if self.conv_type == 'partial':
            inpainted = c11[0].transpose(1, 2) * (1 - masks)
        else:
            inpainted = c11.transpose(1, 2) * (1 - masks)

        out = inpainted + masked_imgs
        return out

    def forward(self, masked_imgs, masks, guidances=None):
        # B, L, C, H, W = masked.shape
        inp = self.preprocess(masked_imgs, masks, guidances)

        encoded_features = self.downsample_module(inp)

        c11 = self.upsample_module(encoded_features)

        out = self.postprocess(masked_imgs, masks, c11)

        return out


class Generator(nn.Module):
    def __init__(
        self, nc_in, nc_out, nf, use_bias, norm, conv_dim, conv_type, use_refine=False
    ):
        super().__init__()
        self.coarse_net = CoarseNet(
            nc_in, nc_out, nf, use_bias, norm, conv_dim, conv_type
        )

    def forward(self, masked_imgs, masks, guidances=None):
        coarse_outputs = self.coarse_net(masked_imgs, masks, guidances)
        return {"outputs": coarse_outputs}


##################
# Discriminators #
##################
