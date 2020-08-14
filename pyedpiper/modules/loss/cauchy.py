import torch

from .functional import cauchy_loss
from .loss import Loss

__all__ = ["CauchyLoss"]


class CauchyLoss(Loss):
    __constants__ = ['c', 'reduction']

    def __init__(self, c=1.0, reduction='mean', ignore_index=None):
        super(CauchyLoss, self).__init__(reduction=reduction)
        self.c = c
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target[mask]
            input = input[mask]

        if not len(target):
            return torch.tensor(0.).to(input.device)

        return cauchy_loss(input, target.float(), self.c, self.reduction)
