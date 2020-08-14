from torch import nn

from .reduction import legacy_get_string


class Loss(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
