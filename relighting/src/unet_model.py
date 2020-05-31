""" Full assembly of the parts to form the complete network """

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, depth=32, normalization=None, device='cuda'):
        super(UNet, self).__init__()
        self.device = device
        self.n_channels = n_channels
        self.inc = DoubleConv(n_channels, depth, normalization=normalization, device=device)
        self.down1 = Down(depth, depth * 2, normalization=normalization, device=device)
        self.down2 = Down(depth * 2, depth * 4, normalization=normalization, device=device)
        self.down3 = Down(depth * 4, depth * 8, normalization=normalization, device=device)
        self.down4 = Down(depth * 8, depth * 16 // 2, normalization=normalization, device=device)

        self.up1 = Up(depth * 16, depth * 8 // 2, device=device)
        self.up2 = Up(depth * 8, depth * 4 // 2, device=device)
        self.up3 = Up(depth * 4, depth * 2 // 2, device=device)
        self.up4 = Up(depth * 2, depth, device=device)
        self.outc = OutConv(depth, 3, device=device)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out
