# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:38:48 2020

@author: nsde
"""
import torch.nn.functional as F

from pytorch_metrics import ClassificationMetric


class Precision(ClassificationMetric):

    def reset(self):
        self._tp = 0.0
        self._fp = 0.0

    def update(self, target, pred):
        self.check_input(target, pred)
        self.check_type(target, pred)
        target, pred = self.transform(target, pred)
        
        target = F.one_hot(target.long(), num_classes=self._num_classes)
        pred = F.one_hot(pred.long(), num_classes=self._num_classes)
        
        self._tp += (target*pred).sum(dim=-1).sum(dim=0)
        self._fp += ((1-target)*pred).sum(dim=-1).sum(dim=0)

    def compute(self):
        val = self._tp / (self._tp + self._fp)
        try:
            return val.item()
        except:
            return val
        