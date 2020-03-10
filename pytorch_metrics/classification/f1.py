# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 09:39:02 2020

@author: nsde
"""

from pytorch_metrics.utils import check_non_zero_sample_size
from .recall import Recall
from .precision import Precision


class F1(Recall, Precision):
    def reset(self):
        super(Recall, self).reset()
        super(Precision, self).reset()

    def update(self, target, pred):
        super(Recall, self).update(target, pred)
        super(Precision, self).update(target, pred)

    def compute(self):
        recall = super(Recall, self).compute()
        precision = super(Precision, self).compute()
        return precision * recall * 2 / (precision + recall)
