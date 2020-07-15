# TODO: read arxiv papers related to each loss and finish up the docstrings accordingly

import math
from typing import Optional

import torch
import torch.functional as F

__all__ = ["focal_loss_with_logits", "cauchy_loss", "wing_loss", "label_smoothed_nll_loss"]


def focal_loss_with_logits(input: torch.Tensor,
                           target: torch.Tensor,
                           gamma: float = 2.0,
                           alpha: Optional[float] = 0.25,
                           reduction="mean",
                           normalized=False,
                           reduced_threshold: Optional[float] = None, ) -> torch.Tensor:
    """Compute binary focal loss between target and output logits.

    Source:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/focal_loss.py

    Args:
        input (_torch.Tensor): Tensor of arbitrary shape
        target (_torch.Tensor): Tensor of the same shape as input
        gamma (float):
        alpha (float):
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    """

    target = target.type(input.type())

    logpt = F.binary_cross_entropy_with_logits(input, target, reduction="none")
    pt = torch.exp(-logpt)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt

    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)

    if normalized:
        norm_factor = focal_term.sum() + 1e-5
        loss /= norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


def cauchy_loss(input: torch.Tensor,
                target: torch.Tensor,
                c: float = 1.0,
                reduction='mean'):

    x = input - target
    loss = torch.log(0.5 * (x / c) ** 2 + 1)

    if reduction == 'sum':
        loss = loss.sum()

    if reduction == 'mean':
        loss = loss.mean()

    return loss


def wing_loss(input: torch.Tensor,
              target: torch.Tensor,
              width=5,
              curvature=0.5,
              reduction="mean") -> torch.Tensor:
    """
    https://arxiv.org/pdf/1711.06753.pdf

    Args:
        input:
        target:
        width:
        curvature:
        reduction:
    Returns:

    """

    diff_abs = (target - input).abs()
    loss = diff_abs.clone()

    idx_smaller = diff_abs < width
    idx_bigger = diff_abs >= width

    loss[idx_smaller] = width * torch.log(1 + diff_abs[idx_smaller] / curvature)

    C = width - width * math.log(1 + width / curvature)
    loss[idx_bigger] = loss[idx_bigger] - C

    if reduction == "sum":
        loss = loss.sum()

    if reduction == "mean":
        loss = loss.mean()

    return loss


def label_smoothed_nll_loss(lprobs: torch.Tensor,
                            target: torch.Tensor,
                            epsilon: float,
                            ignore_index=None,
                            reduction="mean",
                            dim=-1) -> torch.Tensor:
    """
    Source:
        https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py

    Args:
        lprobs: Log-probabilities of predictions (e.g after log_softmax)
        target:
        epsilon:
        ignore_index:
        reduction:
        dim:

    Returns:
        _torch.Tensor: The loss value tensor
    """

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        # nll_loss.masked_fill_(pad_mask, 0.0)
        # smooth_loss.masked_fill_(pad_mask, 0.0)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)

    if reduction == "sum":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == "mean":
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

    eps_i = epsilon / lprobs.size(dim)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

    return loss

