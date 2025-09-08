from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
import torch.nn as nn

class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, groups=groups, stride=stride, padding=padding, bias=False),
            BatchNorm2d(num_features=out_c),
            PReLU(num_parameters=out_c)
        )

    def forward(self, x):
        return self.layers(x)


class LinearBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1,norm=True):
        super(LinearBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            BatchNorm2d(num_features=out_c) if norm else nn.Identity()
        )

    def forward(self, x):
        return self.layers(x)


class DepthWise(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(DepthWise, self).__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            ConvBlock(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1)),
            ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride),
            LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        )

    def forward(self, x):
        short_cut = None
        if self.residual:
            short_cut = x
        x = self.layers(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(DepthWise(c, c, True, kernel, stride, padding, groups))
        self.layers = Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class GDC(Module):
    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        self.layers = nn.Sequential(
            LinearBlock(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0),norm=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            BatchNorm1d(512),
            Linear(512, embedding_size, bias=False),
            BatchNorm1d(embedding_size))

    def forward(self, x):
        return self.layers(x)