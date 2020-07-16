import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["OHEMNLLLoss"]


class OHEMNLLLoss(nn.NLLLoss):
    """Online hard example mining.

    Original from http://www.erogol.com/online-hard-example-mining-pytorch/

    """

    def __init__(self, ratio):
        super().__init__(None, True)
        self.ratio = ratio

    def forward(self, input, target, ratio=None):
        input = F.log_softmax(input, dim=1)

        if ratio is not None:
            self.ratio = ratio

        num_inst = input.size(0)
        num_hns = int(self.ratio * num_inst)
        x_ = input.clone()
        inst_losses = torch.zeros(num_inst).type_as(input)

        for idx, label in enumerate(target.data):
            inst_losses[idx] = -x_.data[idx, label]

        _, idxs = inst_losses.topk(num_hns)
        input_hn = input.index_select(0, idxs)
        target_hn = target.index_select(0, idxs)

        return F.nll_loss(input_hn, target_hn)
