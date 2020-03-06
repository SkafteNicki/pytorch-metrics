# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:37:27 2020

@author: nsde
"""

from pytorch_metrics import Metric

class Accuracy(Metric):
    def reset(self):
        self._tp = 0
        self._n = 0
        
    def update(self, target, pred):
        target, pred = self.transform(target, pred)
        self._tp = (target == pred).sum()
        self._n = target.shape[0]
        
    def compute(self):
        return self._tp / self._n
        