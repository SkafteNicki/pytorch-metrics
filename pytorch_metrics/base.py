# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:41:22 2020

@author: nsde
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from .utils import atleast_2d

class Metric(ABC):
    """
    Base class
    """
    memory_efficient = True
    def __init__(self,
                 transform=lambda x,y: (x,y)):
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
    

class MetricDict(Metric):
    """
    For multiple metrics
    """

    def __init__(self,
                 metric_dict):
        self.metric_dict = metric_dict

        # Initialize metric variables
        self.reset()

    def reset(self):
        [m.reset() for m in self.metric_dict.values()]

    def update(self, target, pred):
        [m.update(target, pred) for m in self.metric_dict.values()]

    def compute(self):
        return {[(k, m.compute()) for k, m in self.metric_dict.items()]}


class RunningAverage(Metric):
    def __init__(self,
                 base_metric,
                 alpha=0.98):
        self.base_metric = base_metric
        self.alpha = alpha

    def reset(self):
        self._running = None
        self.base_metric.reset()

    def update(self, target, pred):
        self.base_metric.update(target, pred)

    def compute(self):
        if self._running is None:
            self._running = self.base_metric.compute()
        else:
            self._running = self._running * self.alpha + \
                (1-self.alpha) * self.base_metric.compute()
        return self._running
    
class BatchedMetric(Metric):
    def __init__(self,
                 base_metric,
                 batch_size,
                 dim=0):
        self.base_metrics = [deepcopy(base_metric) for _ in range(batch_size)]
        self.batch_size = batch_size
        self.dim = dim
        
    def reset(self):
        [m.reset() for m in self.base_metrics()]
        
    def update(self, target, pred):
        for i, m in enumerate(self.base_metrics):
            m.update(target[i], pred[i])
        
    def compute(self):
        return [m.compute for m in self.base_metrics]