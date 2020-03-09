# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:10:16 2020

@author: nsde
"""

import torch
from pytorch_metrics import Metric
from pytorch_metrics.utils import check_non_zero_sample_size


class MeanSquaredError(Metric):
    name = 'meansquarederror'
    memory_efficient = True

    def reset(self):
        self._squared_error = 0
        self._n = 0

    def update(self, target, pred):
        target, pred = self.tobatch(target, pred)
        target, pred = self.transform(target, pred)
        self._squared_error += torch.pow(pred - target, 2).sum(dim=0)
        self._n += target.shape[0]

    def compute(self):
        check_non_zero_sample_size(self._n)
        val = self._squared_error / self._n
        try:
            return val.item()
        except:
            return val
