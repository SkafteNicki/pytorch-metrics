# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:55:14 2020

@author: nsde
"""

import torch
from .accuracy import Accuracy


class FilteredAccuracy(Accuracy):
    """
    FilteredAccuracy can be used to calculate the accuracy on a subset of the
    classes.
    
    Args: 
        transform: transform to apply before calculating metric
        labels_to_ignore: `list` with labels that should be ignored
    
    """
    name = 'filteredaccuracy'
    memory_efficient = True
    
    def __init__(self, transform=None, labels_to_ignore=[]):
        self.labels_to_ignore = labels_to_ignore
        super().__init__(transform, is_multilabel)

    def update(self, target, pred):
        self.check_input(target, pred)
        self.check_type(target, pred)
        target, pred = self.transform(target, pred)

        correct = torch.eq(target, pred)
        for l in self.labels_to_ignore:
            correct[target == l] = 0
            self._n -= (target == l).sum(dim=0)

        self._num_correct += correct.sum(dim=0)
        self._n += target.shape[0]
