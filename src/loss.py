import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import nn.Module
from torch.nn.modules.loss import _WeightedLoss


class PerPixelCrossEntropyLoss(_WeightedLoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None
    ):
        super(PerPixelCrossEntropyLoss, self).__init__(
            weight, size_average, reduce
        )
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor):
        b, c, h, w, = input.shape
        # transform our tensors into a shape that that cross_entropy understands
        input_unraveled = input.view((b, c, h * w))
        target_unraveled = target.view((b, c, h*w))
        # argmax of the channels is the correct label
        target_labels = target_unraveled.argmax(1)

        return F.cross_entropy(
            input_unraveled,
            target_labels,
            weight=self.weight,
            ignore_index=self.ignore_index,
        )

class FeatureLoss(nn.Module):
    def __init__(self, feature_extractor):
        # e.g. VGG
        self.feature_extractor = feature_extractor

    def forward(self, generated, actual):
        generated_feat = self.feature_extractor(generated)
        actual_feat = self.feature_extractor(actual)
        pass

