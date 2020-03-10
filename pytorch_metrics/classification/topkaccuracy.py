# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:37:46 2020

@author: nsde
"""
import torch
from pytorch_metrics import Metric
from pytorch_metrics.utils import check_non_zero_sample_size

class TopkAccuracy(Metric):
    def __init__(self,
                 k=5,
                 transform=lambda x,y: (x,y),
                 device=None):
        self._k = k
        super().__init__(transform, device)
    
    def reset(self):
        self._tp = 0
        self._n = 0
    
    def update(self, target, pred):
        sorted_indices = torch.topk(pred, self._k, dim=1)[1]
        expanded_y = target.view(-1, 1).expand(-1, self._k)
        correct = torch.sum(torch.eq(sorted_indices, expanded_y), dim=1)
        self._tp += torch.sum(correct).item()
        self._n += correct.shape[0]
        
    def compute(self):
        pass