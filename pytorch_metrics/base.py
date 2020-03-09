# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:41:22 2020

@author: nsde
"""
import torch
from abc import ABC, abstractmethod
from copy import deepcopy
from .utils import atleast_2d


class Metric(ABC):
    """
    Base class
    """
    name = 'basemetric'
    memory_efficient = True

    def __init__(self,
                 transform=lambda x, y: (x, y)):
        self.transform = transform

        # Initialize metric variables
        self.reset()

    def __call__(self, target, pred):
        self.reset()
        self.update(target, pred)
        val = self.compute()
        self.reset()
        return val

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, target, pred):
        pass

    @abstractmethod
    def compute(self):
        pass

    def tobatch(self, *tensor):
        return [atleast_2d(t) for t in tensor]

