# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:10:36 2020

@author: nsde
"""

import torch
from pytorch_metrics import Metric
from pytorch_metrics.utils import check_non_zero_sample_size


class MeanSquaredLogarithmicError(Metric):
    name = 'meansquaredlogarithmicerror'
    memory_efficient = True

    def reset(self):
        self._logarithmic_error = 0
        self._n = 0

    def update(self, target, pred):
        assert torch.all(target >= 0), 'All targets needs to be positive'
        target, pred = self.tobatch(target, pred)
        target, pred = self.transform(target, pred)
        self._logarithmic_error += torch.pow(
            (target+1).log() - (pred+1).log(), 2.0).sum(dim=0)
        self._n += target.shape[0]

    def compute(self):
        check_non_zero_sample_size(self._n)
        val = self._logarithmic_error / self._n
        try:
            return val.item()
        except:
            return val
