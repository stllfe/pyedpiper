from functools import partial

from .functional import focal_loss_with_logits
from .loss import Loss

__all__ = ["BinaryFocalLoss", "FocalLoss"]


class BinaryFocalLoss(Loss):
    """
    :param alpha: Prior probability of having positive value in target.
    :param gamma: Power factor for dampening weight (focal strenght).
    :param ignore_index: If not None, targets may contain values to be ignored.
    Target values equal to ignore_index will be ignored from loss computation.
    :param reduced:
    :param threshold:
    """

    def __init__(self,
                 alpha=None,
                 gamma=2,
                 ignore_index=None,
                 reduction="mean",
                 normalized=False,
                 reduced_threshold=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, input, target):
        """Compute focal loss for binary classification problem."""

        target = target.view(-1)
        input = input.view(-1)

        if self.ignore_index is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = target != self.ignore_index
            input = input[not_ignored]
            target = target[not_ignored]

        loss = self.focal_loss_fn(input, target)
        return loss


class FocalLoss(Loss):
    """Focal loss for multi-class problem.

    :param alpha:
    :param gamma:
    :param ignore_index: If not None, targets with given index are ignored
    :param reduced_threshold: A threshold factor for computing reduced focal loss
    """

    def __init__(self,
                 alpha=None,
                 gamma=2,
                 ignore_index=None,
                 reduction="mean",
                 normalized=False,
                 reduced_threshold=None):

        super().__init__()
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, input, target):
        num_classes = input.size(1)
        loss = 0

        # Filter anchors with -1 label from loss computation
        if self.ignore_index is not None:
            not_ignored = target != self.ignore_index

        for cls in range(num_classes):
            cls_label_target = (target == cls).long()
            cls_label_input = input[:, cls, ...]

            if self.ignore_index is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]

            loss += self.focal_loss_fn(cls_label_input, cls_label_target)

        return loss
