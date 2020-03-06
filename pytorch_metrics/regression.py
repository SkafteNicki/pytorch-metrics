# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:10:16 2020

@author: nsde
"""

import torch
from torch import nn

class MeanSquaredError(nn.Module):
    def __init__(self):
        self._values = [ ]
        self._n = 0
    
    def reset(self):
        self._values = [ ]
        self._n = 0
        
    def update(self, pred, target):
        self._values.append(
            torch.mean((pred - target)**2).mean()
            )
        self._n += 1
        
    def compute(self):
        return torch.stack(self._values).mean() / self._n