# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:31:06 2020

@author: nsde
"""

import torch
from pytorch_metrics import Metric
from pytorch_metrics.utils import check_non_zero_sample_size


class MeanAbsoluteError(Metric):
    name = "meanabsoluteerror"
    memory_efficient = True

    def reset(self):
        self._absolut_error = 0
        self._n = 0

    def update(self, target, pred):
        target, pred = self.tobatch(target, pred)
        target, pred = self.transform(target, pred)
        self._absolut_error += torch.abs(pred - target).sum(dim=0)
        self._n += target.shape[0]

    def compute(self):
        check_non_zero_sample_size(self._n)
        val = self._absolut_error / self._n
        try:
            return val.item()
        except:
            return val
