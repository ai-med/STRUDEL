from abc import ABC

from torch.nn import functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss
import torch

from utils.common import normalize


class UBCELoss(_Loss, ABC):
    def forward(self, output, true, uc_map):
        return self._bce_loss(output, true, uc_map)

    @staticmethod
    def _bce_loss(output, true, uc_map):
        bce_loss = F.binary_cross_entropy_with_logits(
            output.float(),
            true.float(),
            reduction='none'
        )
        uc_map = 1 - normalize(uc_map) \
            if not torch.all(uc_map == 0) \
            else 1 - uc_map
        bce_loss = bce_loss * uc_map
        bce_loss = bce_loss.mean(axis=(1, 2, 3))
        return bce_loss.sum() / bce_loss.shape[0]


class DiceLoss(_WeightedLoss, ABC):
    def forward(self, output, target):
        output = torch.sigmoid(output)
        return self._dice_loss_binary(output, target)

    @staticmethod
    def _dice_loss_binary(output, target):
        output = output.view(-1)
        target = target.view(-1)
        intersection = output * target
        numerator = 2 * intersection.sum() + 1.
        denominator = output.sum() + target.sum() + 1.
        return 1 - (numerator / denominator)


class UBCEDiceLoss(_Loss, ABC):
    def __init__(self):
        super(UBCEDiceLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = UBCELoss()

    def forward(self, output, target, unc_map):
        y_2 = self.dice_loss(output, target)
        y_1 = self.bce_loss(output, target, unc_map)
        return y_1 + y_2


class TverskyLoss(_Loss, ABC):
    def forward(self, output, target, alpha=0.7):
        output = torch.sigmoid(output)
        return self._tversky(output, target, alpha)

    @staticmethod
    def _tversky(output, target, alpha):
        tp = (output * target).sum()
        fn = (target * (1 - output)).sum()
        fp = ((1. - target) * output).sum()
        return 1. - (tp + 1.) / (tp + alpha * fn + (1. - alpha) * fp + 1.)
