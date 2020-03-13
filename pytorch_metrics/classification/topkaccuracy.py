# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:37:46 2020

@author: nsde
"""
import torch
from .accuracy import Accuracy


class TopKAccuracy(Accuracy):
    def __init__(self, transform=None, k=5):
        self._k = k
        assert self._k > 1, "k needs to be larger than 1"
        super().__init__(transform)

    def update(self, target, pred):
        self.check_input(target, pred)
        self.check_type(target, pred)
        target, pred = self.transform(target, pred)

        sorted_idx = torch.topk(pred, self._k, dim=-1)[1]
        # TODO: why does this not work

        correct = torch.eq(target.unsqueeze(-1), sorted_idx).sum(dim=-1)

        self._num_correct += correct.sum(dim=0)
        self._n += target.shape[0]

    def check_type(self, target, pred):
        # do not change the default transform
        if pred.shape[-1] < self._k:
            raise RuntimeError(
                "Number of classes needs to be larger than k" "in TopKAccuracy"
            )
        _type = "multiclass"
        _num_classes = pred.shape[-1]

        if self._type is None:
            self._type = _type
            self._num_classes = _num_classes
        else:
            if self._type != _type:
                raise RuntimeError(
                    "Input data type has changed from {0} to {1}".format(
                        self._type, _type
                    )
                )
            if self._num_classes != _num_classes:
                raise RuntimeError(
                    "Input data has changed number of classes "
                    "from {0} to {1}".format(self._num_classes, _num_classes)
                )
