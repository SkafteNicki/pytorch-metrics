# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:34:12 2020

@author: nsde
"""

import torch
from pytorch_metrics import RegressionMetric
from pytorch_metrics.utils import check_non_zero_sample_size


class ExplainedVariance(RegressionMetric):
    name = "explainedvariance"
    memory_efficient = False

    def reset(self):
        self._target = []
        self._pred = []

    def update(self, target, pred):
        self.check_input(target, pred)
        target, pred = self.transform(target, pred)
        self._target.append(target)
        self._pred.append(pred)

    def compute(self):
        check_non_zero_sample_size(len(self._target))
        target = torch.cat(self._target, dim=0)
        pred = torch.cat(self._pred, dim=0)
        val = 1 - (target - pred).var(dim=0) / target.var(dim=0)
        try:
            return val.item()
        except:
            return val
