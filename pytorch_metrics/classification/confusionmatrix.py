# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:39:09 2020

@author: nsde
"""
import torch
from pytorch_metrics import ClassificationMetric
from pytorch_metrics.utils import check_non_zero_sample_size


class ConfusionMatrix(ClassificationMetric):
    name = 'confusionmatrix'
    memory_efficient = False
    
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

        d = target.shape[-1]
        batch_vec = torch.arange(target.shape[-1])
        # this will account for multilabel
        unique_labels = batch_vec * self._num_classes**2 \
            + target * self._num_classes + pred

        bins = torch.bincount(unique_labels, minlength=d*self._num_classes ** 2)
        bins = bins.reshape(d, self._num_classes, self._num_classes).squeeze()
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