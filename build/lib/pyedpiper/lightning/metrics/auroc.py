from typing import Optional, Any

import numpy as np
import torch
from pytorch_lightning.metrics import SklearnMetric


class AUROC(SklearnMetric):
    """
    Compute Area Under the Curve (AUC) from prediction scores

    Note:
        this implementation is restricted to the binary classification task
        or multilabel classification task in label indicator format.

    """

    def __init__(
            self,
            average: Optional[str] = 'macro',
            multi_class: Optional[str] = None,
            reduce_group: Any = torch.distributed.group.WORLD,
            reduce_op: Any = torch.distributed.ReduceOp.SUM,
    ):
        """
        Args:
            average: If None, the scores for each class are returned. Otherwise, this determines the type of
                averaging performed on the data:

                * If 'micro': Calculate metrics globally by considering each element of the label indicator
                  matrix as a label.
                * If 'macro': Calculate metrics for each label, and find their unweighted mean.
                  This does not take label imbalance into account.
                * If 'weighted': Calculate metrics for each label, and find their average, weighted by
                  support (the number of true instances for each label).
                * If 'samples': Calculate metrics for each instance, and find their average.

            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('roc_auc_score',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         average=average,
                         multi_class=multi_class)

    def forward(
            self,
            y_score: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Args:
            y_score: Target scores, can either be probability estimates of the positive class,
                confidence values, or binary decisions.
            y_true: True binary labels in binary label indicators.
            sample_weight: Sample weights.

        Return:
            Area Under Receiver Operating Characteristic Curve
        """
        return super().forward(y_score=y_score, y_true=y_true,
                               sample_weight=sample_weight)
