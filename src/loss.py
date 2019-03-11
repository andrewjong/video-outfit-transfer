from abc import ABC

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


class FeatureLoss(ABC, nn.Module):
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

class L1FeatureLoss(FeatureLoss):
    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)
        self.loss_fn = nn.L1Loss()

    def forward(self, generated, actual):
        generated_feat = self.feature_extractor(generated.detach())
        actual_feat = self.feature_extractor(actual.detach())

        loss = self.loss_fn(generated_feat, actual_feat)
        return loss

class MultiLayerFeatureLoss(nn.Module):
    def __init__(self, feature_extractor, num_layers=3):
        # e.g. VGG
        self.feature_extractor = feature_extractor
        self.layer_weights = [i + 1 / num_layers for i in range(num_layers)]

    def forward(self, generated, actual):
        generated_feat = self.feature_extractor(generated)
        actual_feat = self.feature_extractor(actual)
        # TODO: stuck on how to implement layer weights
        # maybe this will help: https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/2

        pass

