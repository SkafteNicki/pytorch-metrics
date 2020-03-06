# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:31:06 2020

@author: nsde
"""

import torch
from pytorch_metrics import Metric

class MeanAbsoluteError(Metric):
    
    def reset(self):
        self._absolut_error = 0
        self._n = 0
        
    def update(self, target, pred):
        self._absolut_error += torch.abs(pred - target).sum().item()
        self._n += target.shape[0]
        
    def compute(self):
        if self._n == 0:
            raise RuntimeError('Must have one sample, before compute can be called')
        return self._absolut_error / self._n