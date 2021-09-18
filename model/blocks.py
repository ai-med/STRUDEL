import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import OctaveConvBN
from squeeze_and_excitation import squeeze_and_excitation as se


class DoubleOctConv(nn.Module):

    def __init__(self, kshape, in_channels, out_channels, drop_out=0.2, alpha_in=0.5, alpha_out=0.5,
                 se_block_type=None):
        super().__init__()

        if se_block_type == se.SELayer.CSE.value:
            self.SELayer_h = se.ChannelSELayer(int(out_channels * alpha_out))
            self.SELayer_l = se.ChannelSELayer(int(out_channels * alpha_out))

        elif se_block_type == se.SELayer.SSE.value:
            self.SELayer_h = se.SpatialSELayer(int(out_channels * alpha_out))
            self.SELayer_l = se.SpatialSELayer(int(out_channels * alpha_out))

        elif se_block_type == se.SELayer.CSSE.value:
            self.SELayer_h = se.ChannelSpatialSELayer(int(out_channels * alpha_out))
            self.SELayer_l = se.ChannelSpatialSELayer(int(out_channels * alpha_out))

        if alpha_in == 0:
            self.conv1 = OctaveConvBN(in_channels, out_channels, kernel_size=kshape, padding=1,
                                    alpha_in=alpha_in, alpha_out=alpha_out)
            self.conv2 = OctaveConvBN(out_channels, out_channels, kernel_size=kshape, padding=1,
                                    alpha_in=alpha_out, alpha_out=alpha_out)
        else:
            self.conv1 = OctaveConvBN(in_channels, out_channels, kernel_size=kshape, padding=1,
                                   alpha_in=alpha_in, alpha_out=alpha_in)
            self.conv2 = OctaveConvBN(out_channels, out_channels, kernel_size=kshape, padding=1,
                                    alpha_in=alpha_in, alpha_out=alpha_out)
        self.act = nn.ReLU(inplace=True)
        if drop_out > 0:
            self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x_h, x_l = self.conv1(x)
        x_h = self.act(x_h)
        x_l = self.act(x_l)
        x_h, x_l = self.conv2((x_h, x_l))
        x_h = self.act(x_h)
        x_l = self.act(x_l)

        if hasattr(self, 'SELayer_h') and hasattr(self, 'SELayer_l'):
            x_h = self.SELayer_h(x_h)
            x_l = self.SELayer_l(x_l)

        if hasattr(self, 'dropout'):
            x_h = self.dropout(x_h)
            x_l = self.dropout(x_l)

        return x_h, x_l


class DownOct(nn.Module):

    def __init__(self, in_channels, out_channels, drop_out=0.2, alpha_in=0.5, alpha_out=0.5, se_block_type=None):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleOctConv(kshape=3, in_channels=in_channels, out_channels=out_channels, drop_out=drop_out,
                                  alpha_in=alpha_in, alpha_out=alpha_out, se_block_type=se_block_type)

    def forward(self, x):
        x_h, x_l = x
        x_h = self.pool(x_h)
        x_l = self.pool(x_l)
        x_h, x_l = self.conv((x_h, x_l))

        return x_h, x_l


class UpOct(nn.Module):

    def __init__(self, in_channels, out_channels, drop_out=0.2, alpha_in=0.5, alpha_out=0.5, se_block_type=None):
        super().__init__()
        self.upsample_h = nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2)
        self.upsample_l = nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2)
        self.conv = DoubleOctConv(kshape=3, in_channels=in_channels, out_channels=out_channels, drop_out=drop_out,
                                  alpha_in=alpha_in, alpha_out=alpha_out, se_block_type=se_block_type)

    def forward(self, x1, x2):
        x1_h, x1_l = x1
        x1_h = self.upsample_h(x1_h)
        x1_l = self.upsample_l(x1_l)

        x2_h, x2_l = x2

        return self.conv((torch.cat([x2_h, x1_h], dim=1), torch.cat([x2_l, x1_l], dim=1)))


class OutOctConv(nn.Module):

    def __init__(self, in_channels, out_channels, alpha_in=0.5, alpha_out=0):
        super(OutOctConv, self).__init__()
        self.conv = OctaveConvBN(in_channels, out_channels, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out)

    def forward(self, x):
        return self.conv(x)
