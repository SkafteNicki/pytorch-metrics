# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 12:07:33 2020

@author: nsde
"""

import torch
from pytorch_metrics import Metric
from pytorch_metrics.utils import check_non_zero_sample_size


class R2Score(Metric):
    name = 'r2score'
    memory_efficient = False

    def reset(self):
        self._target = []
        self._pred = []

    def update(self, target, pred):
        target, pred = self.tobatch(target, pred)
        target, pred = self.transform(target, pred)
        self._target.append(target)
        self._pred.append(pred)

    def compute(self):
        check_non_zero_sample_size(len(self._target))
        target = torch.cat(self._target, dim=0)
        pred = torch.cat(self._pred, dim=0)
        ss_res = torch.pow(target - pred, 2.0).sum(dim=0)
        ss_tot = torch.pow(target - target.mean(dim=0), 2.0).sum(dim=0)
        return 1 - ss_res / ss_tot
