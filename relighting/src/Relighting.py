import torch.nn as nn
from src import unet_model
from src import relighting_model
import copy

class Relighting(nn.Module):
    def __init__(self, device='cuda', task=None):
        super(Relighting, self).__init__()
        self.task = task
        self.device = device
        if self.task == 'normalization':
            self.norm_net = unet_model.UNet(n_channels=3, normalization='instanceNorm', device=self.device)
        else:
            self.norm_net = unet_model.UNet(n_channels=3, normalization='instanceNorm', device=self.device)
            self.relighting_net = relighting_model.RelightingNet(n_channels=3, normalization='batchNorm',
                                                                 task=task, device=self.device)

    def load_normalization_net(self, norm_net):
        self.norm_net = copy.deepcopy(norm_net)
        for param in self.norm_net.parameters():
            param.requires_grad = False

    def forward(self, x, t=None):
        if self.task == 'normalization':
            norm_x = self.norm_net(x)
            return norm_x
        elif self.task == 'relighting_one_to_one':
            norm_x = self.norm_net(x)
            return self.relighting_net(x, y=norm_x)
        elif self.task == 'relighting_one_to_any':
            if t is None:
                raise Exception('No guide image(s) are given')
            norm_x = self.norm_net(x)
            return self.relighting_net(x, y=norm_x, z=t)
        else:
            raise Exception(f'Wrong task value; the given value is {self.task}')


