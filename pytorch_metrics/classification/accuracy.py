# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:37:27 2020

@author: nsde
"""

import torch
from pytorch_metrics import ClassificationMetric
from pytorch_metrics.utils import check_non_zero_sample_size


class Accuracy(ClassificationMetric):
    name = 'accuracy'
    memory_efficient = True
    
    def reset(self):
        self._num_correct = 0.0
        self._n = 0
        super().reset()

    def update(self, target, pred):
        self.check_input(target, pred)
        self.check_type(target, pred)
        target, pred = self.transform(target, pred)

        correct = torch.eq(target, pred)
        self._num_correct += correct.sum(dim=0)
        self._n += target.shape[0]

    def compute(self):
        check_non_zero_sample_size(self._n)
        val = self._num_correct / self._n
        try:
            return val.item()
        except:
            return val
