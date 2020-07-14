import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAvgPool2d(nn.Module):
    """Global average pooling over the input's spatial dimensions"""

    def __init__(self, flatten=True):
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        if self.flatten:
            x = x.view(x.size(0), x.size(1))
        return x


class GlobalMaxPool2d(nn.Module):
    """Global max pooling over the input's spatial dimensions"""

    def __init__(self, flatten=False):
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        x = F.adaptive_max_pool2d(x, output_size=(1, 1))
        if self.flatten:
            x = x.view(x.size(0), x.size(1))
        return x


class GlobalMixStackPool2d(nn.Module):
    """Global max pooling and avg pooling stacked together."""

    def __init__(self, flatten=True):
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        x_avg = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x_max = F.adaptive_max_pool2d(x, output_size=(1, 1))
        x = torch.cat((x_avg, x_max), dim=1)
        if self.flatten:
            x = x.view(x.size(0), x.size(1))
        return x


class GlobalAvgMeanStdStackPool2d(nn.Module):
    """Global average pooling with its mean and std stacked horizontally."""

    def __init__(self, flatten=True):
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = torch.cat((x.mean(dim=1), x.std(dim=1)), dim=1)
        if self.flatten:
            x = x.view(x.size(0), x.size(1))
        return x
