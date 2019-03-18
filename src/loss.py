from abc import ABC

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss


class PerPixelCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None):
        super(PerPixelCrossEntropyLoss, self).__init__(weight, size_average, reduce)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor):
        b, c, h, w, = input.shape
        # transform our tensors into a shape that that cross_entropy understands
        input_unraveled = input.view((b, c, h * w))
        target_unraveled = target.view((b, c, h * w))
        # argmax of the channels is the correct label
        target_labels = target_unraveled.argmax(1)

        return F.cross_entropy(
            input_unraveled,
            target_labels,
            weight=self.weight,
            ignore_index=self.ignore_index,
        )


class FeatureLoss(ABC, nn.Module):
    def __init__(self, feature_extractor, scale=224/512):
        super().__init__()
        # set to eval mode to disable dropout and such
        self.feature_extractor = feature_extractor.eval()
        self.scale = scale

    def downsize(self, *inputs):
        outs = []
        for a in inputs:
            outs.append(nn.functional.interpolate(a, scale_factor=self.scale))

        return tuple(outs)


class L1FeatureLoss(FeatureLoss):
    def __init__(self, feature_extractor, scale):
        super().__init__(feature_extractor, scale)
        self.loss_fn = nn.L1Loss()

    def forward(self, generated, actual):
        generated, actual = self.downsize(generated, actual)
        print(generated.shape, actual.shape)
        generated_feat = self.feature_extractor(generated.detach())
        actual_feat = self.feature_extractor(actual.detach())

        loss = self.loss_fn(generated_feat, actual_feat)
        return loss


class MultiLayerFeatureLoss(FeatureLoss):
    """
    Computes the feature loss with the last n layers of a deep feature extractor.
    """
    def __init__(self, feature_extractor, scale, loss_fn=nn.L1Loss(), num_layers=3):
        """

        :param feature_extractor: an pretrained model, i.e. resnet18(), vgg19()
        :param loss_fn: an initialized loss function
        :param num_layers: number of layers from the end to keep. e.g. 3 will compute
        the loss using the last 3 layers of the feature extractor network
        """
        # e.g. VGG
        super().__init__(feature_extractor, scale)

        features = list(feature_extractor.features)
        self.num_layers = num_layers
        self.loss_fn = loss_fn

        self.layer_weights = [i + 1 / num_layers for i in range(num_layers)]

        self.features = nn.ModuleList(features).eval()

        start = len(self.features) - num_layers
        end = len(self.features)
        self.layers_to_keep = {i for i in range(start, end)}

    def extract_intermediate_layers(self, x):
        """
        Extracts features of intermediate layers using the feature extractor
        :param x: the input
        :return:
        """
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.layers_to_keep:
                results.append(x)

        return results

    def forward(self, generated, actual):
        generated, actual = self.downsize(generated, actual)
        generated_feat_list = self.extract_intermediate_layers(generated)
        actual_feat_list = self.extract_intermediate_layers(actual)
        total_loss = 0

        for i, w in enumerate(self.layer_weights):
            total_loss += w * self.loss_fn(generated_feat_list[i], actual_feat_list[i])

        return total_loss
