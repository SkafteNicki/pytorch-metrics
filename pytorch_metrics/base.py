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


class MetricCollection(Metric):
    """
    For multiple metrics
    """

    def __init__(self, metrics):
        if isinstance(metrics, dict):
            self.metrics = metrics
        elif isinstance(metrics, list):
            self.metrics = dict((m.name, m) for m in metrics)
        else:
            raise ValueError('Expected input to MetricCollection, to either be'
                             ' a dist or list of metrics. Got {0}'.format(metrics))
        for i, m in enumerate(self.metrics.values()):
            assert isinstance(m, Metric), 'Metric {0} passed to MetricCollection'\
                ' is not a valid Metric'
        
        # Initialize metric variables
        self.reset()

    def reset(self):
        [m.reset() for m in self.metrics.values()]

    def update(self, target, pred):
        [m.update(target, pred) for m in self.metrics.values()]

    def compute(self):
        return dict((k, m.compute()) for k, m in self.metrics.items())


class RunningAverage(Metric):
    def __init__(self,
                 base_metric,
                 alpha=0.98):
        self.base_metric = base_metric
        self.alpha = alpha

        self.reset()

    def __call__(self, target, pred):
        raise ValueError('Single evaluation cannot be used when metric is '
                         'wrapped in RunningAverage')

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


class Reduce(Metric):
    def __init__(self,
                 base_metric,
                 reduction=torch.sum):
        self.base_metric = base_metric
        self.reduction = reduction
        # TODO: assert that self.reduction is a callable

    def compute(self):
        val = self.base_metric.compute()
        try:
            return torch.sum(val)  # tensor
        except:
            return val  # scalar
