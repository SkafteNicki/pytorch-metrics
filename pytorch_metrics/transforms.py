# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:48:38 2020

@author: nsde
"""
from abc import ABC, abstractmethod


class Transform(ABC):
    def __init__(self):
        pass

    def __call__(self, target, pred):
        output = self.forward(target, pred)
        assert len(output), "Transform should return two values: (target, pred)"
        return output

    @abstractmethod
    def forward(self, target, pred):
        pass


class SequentialTransform(Transform):
    def __init__(self, *transforms):
        self.transforms = transforms

    def forward(self, target, pred):
        for t in self.transforms:
            target, pred = t(target, pred)


class DefaultTransform(Transform):
    def forward(self, target, pred):
        return target, pred


class ArgmaxTransform(Transform):
    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, target, pred):
        return target, pred.argmax(dim=self.dim)


class RoundTransform(Transform):
    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, target, pred):
        return target, pred.round(dim=self.dim)

class ROCTransform(Transform):
    def __init__(self, line):
        self.line = line
        
    def forward(self, target, pred):
        if pred.shape[-1] == 2:
            pred = pred[:,:,1:] # reduce to single dimension

        pred = (pred>self.line.type_as(pred)).long()
        return target, pred