""" Parts of the U-Net model """

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, normalization=None, device='cuda'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if normalization == 'instanceNorm':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif normalization == 'batchNorm':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        self.double_conv.to(device=device)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, normalization=None, device='cuda', task=None):
        super().__init__()
        if task is None:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels, normalization=normalization)
            )
        elif task == 'relighting_one_to_one':
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels * 2, out_channels, normalization=normalization)
            )
        elif task == 'relighting_one_to_any':
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels * 3, out_channels, normalization=normalization)
            )
        self.maxpool_conv.to(device=device)

    def forward(self, x, y=None, z=None):
        if y is None and z is None:
            return self.maxpool_conv(x)
        elif z is None:
            return self.maxpool_conv(torch.cat([x, y], dim=1))
        else:
            return self.maxpool_conv(torch.cat([x, y, z], dim=1))

class Merging(nn.Module):
    """double conv"""

    def __init__(self, in_channels, out_channels, normalization=None, device='cuda', task=None):
        super().__init__()
        if task == 'relighting_one_to_one':
            self.dconv = DoubleConv(in_channels * 2, out_channels, normalization=normalization)
            self.dconv.to(device=device)

        if task == 'relighting_one_to_any':
            self.dconv = DoubleConv(in_channels * 3, out_channels, normalization=normalization)
            self.dconv.to(device=device)

    def forward(self, x, y, z=None):
        if z is None:
            return self.dconv(torch.cat([x, y], dim=1))
        else:
            return self.dconv(torch.cat([x, y, z], dim=1))


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, device='cuda'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        self.up.to(device=device)
        self.conv.to(device=device)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, device='cuda'):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv.to(device=device)

    def forward(self, x):
        return self.conv(x)
