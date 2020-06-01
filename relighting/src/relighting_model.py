""" Full assembly of the parts to form the complete network """

from .unet_parts import *

class RelightingNet(nn.Module):
    def __init__(self, n_channels, depth=32, normalization=None, task=None, device='cuda'):
        super(RelightingNet, self).__init__()
        self.device = device
        self.n_channels = n_channels
        self.task = task
        self.inc = DoubleConv(n_channels, depth, normalization=normalization, device=device)
        self.down1 = Down(depth, depth * 2, normalization=normalization, device=device, task=task)
        self.down2 = Down(depth * 2, depth * 4, normalization=normalization, device=device, task=task)
        self.down3 = Down(depth * 4, depth * 8, normalization=normalization, device=device, task=task)
        self.down4 = Down(depth * 8, depth * 16 // 2, normalization=normalization, device=device, task=task)
        self.inc_norm = DoubleConv(n_channels, depth, normalization=normalization, device=device)
        self.down1_norm = Down(depth, depth * 2, normalization=normalization, device=device)
        self.down2_norm = Down(depth * 2, depth * 4, normalization=normalization, device=device)
        self.down3_norm = Down(depth * 4, depth * 8, normalization=normalization, device=device)
        self.down4_norm = Down(depth * 8, depth * 16 // 2, normalization=normalization, device=device)
        if task == 'relighting_one_to_any':
            self.inc_t = DoubleConv(n_channels, depth, normalization=normalization, device=device)
            self.down1_t = Down(depth, depth * 2, normalization=normalization, device=device)
            self.down2_t = Down(depth * 2, depth * 4, normalization=normalization, device=device)
            self.down3_t = Down(depth * 4, depth * 8, normalization=normalization, device=device)
            self.down4_t = Down(depth * 8, depth * 16 // 2, normalization=normalization, device=device)
        self.merging = Merging(depth * 16//2, depth * 16 // 2, normalization=normalization, device=device, task=task)
        self.up1 = Up(depth * 16, depth * 8 // 2, device=device)
        self.up2 = Up(depth * 8, depth * 4 // 2, device=device)
        self.up3 = Up(depth * 4, depth * 2 // 2, device=device)
        self.up4 = Up(depth * 2, depth, device=device)
        self.outc = OutConv(depth, 3, device=device)

    def forward(self, x, y, z=None):

        x1 = self.inc(x)

        y1 = self.inc(y)
        y2 = self.down1_norm(y1)
        y3 = self.down2_norm(y2)
        y4 = self.down3_norm(y3)
        y5 = self.down4_norm(y4)

        if z is not None:
            z1 = self.inc(z)
            z2 = self.down1_t(z1)
            z3 = self.down2_t(z2)
            z4 = self.down3_t(z3)
            z5 = self.down4_t(z4)

            x2 = self.down1(x1, y=y1, z=z1)
            x3 = self.down2(x2, y=y2, z=z2)
            x4 = self.down3(x3, y=y3, z=z3)
            x5 = self.down4(x4, y=y4, z=z4)
            x6 = self.merging(x5, y=y5, z=z5)
        else:
            x2 = self.down1(x1, y=y1)
            x3 = self.down2(x2, y=y2)
            x4 = self.down3(x3, y=y3)
            x5 = self.down4(x4, y=y4)
            x6 = self.merging(x5, y=y5)

        x = self.up1(x6, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        #out = x * self.outc(x10)
        out = self.outc(x)
        return out
