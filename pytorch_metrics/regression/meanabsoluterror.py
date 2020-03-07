# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:31:06 2020

@author: nsde
"""

import torch
from pytorch_metrics import Metric
from pytorch_metrics.utils import (sync_all_reduce, 
                                   set_is_reduced,
                                   check_non_zero_sample_size)

class MeanAbsoluteError(Metric):
    
    @set_is_reduced
    def reset(self):
        self._absolut_error = 0
        self._n = 0
        
    @set_is_reduced
    def update(self, target, pred):
        target, pred = self.transform(target, pred)
        self._absolut_error += torch.abs(pred - target.view_as(pred)).sum().item()
        self._n += target.shape[0]
        
    @sync_all_reduce('_absolut_error', '_n')
    def compute(self):
        check_non_zero_sample_size(self._n)
        return self._absolut_error / self._n
