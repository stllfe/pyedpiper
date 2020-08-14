from .functional import wing_loss
from .loss import Loss

__all__ = ["WingLoss"]


class WingLoss(Loss):
    __constants__ = ["width", "reduction", "curvature"]

    def __init__(self, width=5, curvature=0.5, reduction="mean"):
        super(WingLoss, self).__init__(reduction=reduction)
        self.width = width
        self.curvature = curvature

    def forward(self, prediction, target):
        return wing_loss(prediction, target, self.width, self.curvature, self.reduction)
