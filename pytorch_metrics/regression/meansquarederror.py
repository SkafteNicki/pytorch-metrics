# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:10:16 2020

@author: nsde
"""

import torch
from pytorch_metrics import Metric
from pytorch_metrics.utils import (sync_all_reduce, set_is_reduced)


class MeanSquaredError(Metric):
    
    def __call__(self, target, pred):
        return torch.pow(target - pred, 2).sum().item()
    
    @set_is_reduced
    def reset(self):
        self._squared_error = 0
        self._n = 0
    
    @set_is_reduced
    def update(self, target, pred):
        self._squared_error += self(target, pred)
        self._n += target.shape[0]

    @sync_all_reduce('_squared_error', '_n')
    def compute(self):
        if self._n == 0:
            raise RuntimeError(
                'Must have one sample, before compute can be called')
        return self._squared_error / self._n
