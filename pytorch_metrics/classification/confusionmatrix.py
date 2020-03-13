# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:39:09 2020

@author: nsde
"""
import torch
from pytorch_metrics import ClassificationMetric
from pytorch_metrics.utils import check_non_zero_sample_size


class ConfusionMatrix(ClassificationMetric):
    def __init__(self, transform=None, is_multilabel=False, normalize=False):
        super().__init__(transform, is_multilabel)
        self.normalize = normalize

    def reset(self):
        self.confmat = 0.0
        self._n = 0
        super().reset()

    def update(self, target, pred):
        self.check_input(target, pred)
        self.check_type(target, pred)
        target, pred = self.transform(target, pred)

        bins = torch.bincount(
            target * self._num_classes + pred, minlength=self._num_classes ** 2
        )
        bins = bins.reshape(self._num_classes, self._num_classes)
        if self.confmat is None:
            self.confmat = bins
        else:
            self.confmat += bins

        self._n += target.shape[0]

    def compute(self):
        check_non_zero_sample_size(self._n)
        if self.normalize:
            return self.confmat / self._n
        else:
            return self.confmat

    def check_input(self, target, pred):
        ts = target.shape
        ps = pred.shape

        assert (
            target.squeeze().shape == pred.squeeze().shape[:-1]
        ), "Target can only differ in the last dimension"

        # For confmat we cannot support multiple outputs
        # (due to torch.bincount only working with 1D tensors), so only expand
        # target to [N,] and preds to [N,C]
        if len(ts) < 1:
            target.unsqueeze_(0)

        if len(ps) < 1:
            pred.unsqueeze_(0)

        if len(ps) < 2:
            pred.unsqueeze_(0)
