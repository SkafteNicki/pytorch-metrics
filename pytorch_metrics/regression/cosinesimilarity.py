# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:42:38 2020

@author: nsde
"""

import torch
from pytorch_metrics import Metric
from pytorch_metrics.utils import check_non_zero_sample_size

class CosineSimilarity(Metric):
    name = 'cosinesimilarity'
    memory_efficient = True
    
    def reset(self):
        self._dot_xy = 0
        self._norm_x = 0
        self._norm_y = 0
        self._n = 0
    
    def update(self, target, pred):
        target, pred = self.transform(target, pred)
        self._dot_xy += (target * pred).sum(dim=0)
        self._norm_x += torch.pow(target, 2.0).sum(dim=0)
        self._norm_y += torch.pow(pred, 2.0).sum(dim=0)
        self._n += target.shape[0]

    def compute(self):
        check_non_zero_sample_size(self._n)
        val = self._dot_xy / (self._norm_x.sqrt() * self._norm_y.sqrt())
        try:
            return val.item()
        except:
            return val
