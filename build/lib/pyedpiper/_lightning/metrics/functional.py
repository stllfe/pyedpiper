import torch

from pytorch_lightning.metrics.functional import stat_scores_multiple_classes, stat_scores
from pytorch_lightning.metrics.functional.classification import get_num_classes
from pytorch_lightning.metrics.functional.reduction import reduce
from typing import Optional


__all__ = ["balanced_accuracy", "sensitivity_specificity"]


def sensitivity_specificity(pred: torch.Tensor,
                            target: torch.Tensor,
                            num_classes: Optional[int] = None,
                            reduction: str = 'elementwise_mean'):
    num_classes = get_num_classes(pred=pred, target=target, num_classes=num_classes)

    if num_classes > 2:
        tps, fps, tns, fns, sups = stat_scores_multiple_classes(pred=pred, target=target, num_classes=num_classes)
    else:
        tps, fps, tns, fns, sups = stat_scores(pred=pred, target=target, class_index=1)

    tps = tps.to(torch.float)
    fps = fps.to(torch.float)
    fns = fns.to(torch.float)
    tns = tns.to(torch.float)

    sensitivity = tps / (fns + tps)
    specificity = tns / (fps + tns)

    # drop NaN after zero division
    sensitivity[sensitivity != sensitivity] = 0
    specificity[specificity != specificity] = 0

    sensitivity = reduce(sensitivity, reduction)
    specificity = reduce(specificity, reduction)

    return sensitivity, specificity


def balanced_accuracy(pred: torch.Tensor,
                      target: torch.Tensor,
                      num_classes: Optional[int] = None,
                      reduction: str = 'elementwise_mean'):

    sens_spec = sensitivity_specificity(pred=pred,
                                        target=target,
                                        num_classes=num_classes,
                                        reduction=reduction)

    return torch.as_tensor(sens_spec).mean()
