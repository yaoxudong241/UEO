from model import common
from model.fat_step1_weight import FAT
import torch.nn as nn
import torch
import random

def make_model(args, parent=False):
    return EDSR_two(args)


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class EDSR_two(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR_two, self).__init__()
        self.EDSR_var = FAT()
        self.EDSR_U = FAT()
        # if args.pre_train_step1 != '.':
        self.EDSR_var.load_state_dict(torch.load(args.pre_train_step1), strict=True)
        self.EDSR_U.load_state_dict(torch.load(args.pre_train_step1), strict=True)

    def forward(self, x):

        with torch.no_grad():
            var = self.EDSR_var(x)
        x = self.EDSR_U(x)
        x = x[0]
        return [x, var[1]]

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
