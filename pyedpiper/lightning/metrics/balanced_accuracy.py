from typing import Any
from typing import Optional

import torch
from pytorch_lightning.metrics import TensorMetric

from .functional import balanced_accuracy


class BalancedAccuracy(TensorMetric):
    """
    Computes the accuracy classification score

    Example:

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = BalancedAccuracy()
        >>> metric(pred, target)
        tensor(0.7500)

    """

    def __init__(
            self,
            num_classes: Optional[int] = None,
            reduction: str = 'elementwise_mean',
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        """
        Args:
            num_classes: number of classes
            reduction: a method for reducing accuracies over labels (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction
        """
        super().__init__(name='balanced_accuracy',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: ground truth labels

        Return:
            A Tensor with the classification score.
        """
        return balanced_accuracy(pred=pred, target=target,
                                 num_classes=self.num_classes, reduction=self.reduction)
