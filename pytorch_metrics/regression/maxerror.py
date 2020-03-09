# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:11:13 2020

@author: nsde
"""

import torch
from pytorch_metrics import Metric


class MaxError(Metric):
    name = 'maxerror'
    memory_efficient = True

    def reset(self):
        self._max_error = torch.tensor([-float('inf')])

    def update(self, target, pred):
        target, pred = self.tobatch(target, pred)
        target, pred = self.transform(target, pred)
        self._max_error = torch.max(torch.abs(target - pred).max(dim=0)[0],
                                    self._max_error.type_as(target))

    def compute(self):
        val = self._max_error
        try:
            return val.item()
        except:
            return val
