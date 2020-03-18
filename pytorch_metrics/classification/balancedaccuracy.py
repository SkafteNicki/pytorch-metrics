# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:41:47 2020

@author: nsde
"""
import torch.nn.functional as F
from .recall import Recall


class BalancedAccuracy(Recall):
    name = 'balancedaccuracy'
    memory_efficient = True
    
    def update(self, target, pred):
        self.check_input(target, pred)
        self.check_type(target, pred)
        target, pred = self.transform(target, pred)

        target = F.one_hot(target.long(), num_classes=self._num_classes)
        pred = F.one_hot(pred.long(), num_classes=self._num_classes)

        self._tp += (target * pred).sum(dim=0)
        self._fn += (target * (1 - pred)).sum(dim=0)

    def compute(self):
        val = (self._tp / (self._tp + self._fn)).mean(dim=-1)
        try:
            return val.item()
        except:
            return val
