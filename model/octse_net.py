from torch import optim

from model.blocks import *


class OctaveSENet(nn.Module):
    def __init__(self, params):
        super(OctaveSENet, self).__init__()

        in_channels = params['IN_CHANNELS']
        in_filters = params['NUM_FILTERS']
        se_block_type = params['SE_BLOCK_TYPE']
        kernel_size = params['KERNEL_SIZE']
        drop_out = params['DROP_OUT']

        self.conv = DoubleOctConv(kshape=kernel_size, in_channels=in_channels, out_channels=in_filters,
                                  drop_out=drop_out, alpha_in=0, alpha_out=0.5)
        self.down1 = DownOct(in_filters, in_filters * 2, drop_out=drop_out, se_block_type=se_block_type)
        self.down2 = DownOct(in_filters * 2, in_filters * 4, drop_out=drop_out, se_block_type=se_block_type)
        self.down3 = DownOct(in_filters * 4, in_filters * 8, drop_out=drop_out, se_block_type=se_block_type)
        self.down4 = DownOct(in_filters * 8, in_filters * 16, drop_out=drop_out, se_block_type=se_block_type)
        self.up1 = UpOct(in_filters * 16, in_filters * 8, drop_out=drop_out, se_block_type=se_block_type)
        self.up2 = UpOct(in_filters * 8, in_filters * 4, drop_out=drop_out, se_block_type=se_block_type)
        self.up3 = UpOct(in_filters * 4, in_filters * 2, drop_out=drop_out, se_block_type=se_block_type)
        self.up4 = UpOct(in_filters * 2, in_filters, drop_out=drop_out, se_block_type=se_block_type)
        self.out_conv = OutOctConv(in_filters, in_filters // in_filters, alpha_in=0.5, alpha_out=0)

    def forward(self, x):

        x1 = self.conv.forward(x)
        x2 = self.down1.forward(x1)
        x3 = self.down2.forward(x2)
        x4 = self.down3.forward(x3)
        x5 = self.down4.forward(x4)
        x = self.up1.forward(x5, x4)
        x = self.up2.forward(x, x3)
        x = self.up3.forward(x, x2)
        x = self.up4.forward(x, x1)
        x, _ = self.out_conv.forward(x)

        return x

    def enable_dropout(self):
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def reset_parameters(self):
        for m in self.modules():
            if m.__class__.__name__.startswith('Conv'):
                m.reset_parameters()
            if m.__class__.__name__.startswith('Linear'):
                m.reset_parameters()

    def outconv_grad(self, requires: bool):
        for m in self.modules():
            if m.__class__.__name__.startswith('Out'):
                for param in m.parameters():
                    param.requires_grad = requires

    def get_optimizer(self, learning_rate: float):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        return optimizer
