from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor

from .functional import label_smoothed_nll_loss

__all__ = ["SmoothCrossEntropyLoss"]


class SmoothCrossEntropyLoss(nn.Module):
    """Drop-in replacement for nn.CrossEntropyLoss that supports label smoothing."""

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(self,
                 reduction="mean",
                 smooth_factor: Optional[float] = None,
                 ignore_index: Optional[int] = None,
                 dim=1):

        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        log_prob = F.log_softmax(input, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            target,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )
