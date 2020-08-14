from typing import Any
from typing import Optional
from typing import Sequence

import torch
from pytorch_lightning.metrics import TensorMetric

from .functional import auroc_binary


class AUROCBinary(TensorMetric):
    """
    Computes the area under curve (AUC) of the receiver operator characteristic (ROC).

    Example:

        >>> pred = torch.Tensor([[0.6, 0.4], [0.5, 0.5], [0.4, 0.6], [0.3, 0.7]])
        >>> target = torch.tensor([0, 1, 0, 1])
        >>> metric = AUROCBinary()
        >>> metric(pred, target)
        tensor(0.7500)

    """

    def __init__(
            self,
            pos_index: int = 1,
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        """
        Args:
            pos_index: positive label index to use in case of multiple dimensions
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction
        """
        super().__init__(name='auroc',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)

        self.pos_index = pos_index

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            sample_weight: Optional[Sequence] = None
    ) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: groundtruth labels
            sample_weight: the weights per sample

        Return:
            torch.Tensor: classification score
        """

        return auroc_binary(pred=pred,
                            target=target,
                            pos_index=self.pos_index,
                            sample_weight=sample_weight)
