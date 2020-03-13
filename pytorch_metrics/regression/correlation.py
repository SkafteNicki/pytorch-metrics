# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:41:17 2020

@author: nsde
"""

import torch
from pytorch_metrics import RegressionMetric
from pytorch_metrics.utils import check_non_zero_sample_size

class Correlation(RegressionMetric):
    name = "correlation"
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
        target_mean = target.mean(dim=0, keepdim=True)
        pred_mean = pred.mean(dim=0, keepdim=True)
        val1 = ((target-target_mean) * (pred-pred_mean)).sum(dim=0)
        val2 = torch.pow(target-target_mean, 2.0).sum(dim=0).sqrt()
        val3 = torch.pow(pred-pred_mean, 2.0).sum(dim=0).sqrt()
        val = val1 / (val2*val3)
        try:
            return val.item() # scalar
        except:
            return val # tensor
        