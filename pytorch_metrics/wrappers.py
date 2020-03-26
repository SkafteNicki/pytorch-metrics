# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 08:18:51 2020

@author: nsde
"""

import torch
from .base import Metric


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
            raise ValueError(
                "Expected input to MetricCollection, to either be"
                " a dist or list of metrics. Got {0}".format(metrics)
            )
        for i, m in enumerate(self.metrics.values()):
            assert isinstance(m, Metric), (
                "Metric {0} passed to MetricCollection" " is not a valid Metric"
            )

        # Initialize metric variables
        self.reset()

    def reset(self):
        [m.reset() for m in self.metrics.values()]

    def update(self, target, pred):
        [m.update(target, pred) for m in self.metrics.values()]

    def compute(self):
        return dict((k, m.compute()) for k, m in self.metrics.items())


class RunningAverage(Metric):
    def __init__(self, base_metric, alpha=0.98):
        self.base_metric = base_metric
        self.alpha = alpha

        self.reset()

    def __call__(self, target, pred):
        raise ValueError(
            "Single evaluation cannot be used when metric is "
            "wrapped in RunningAverage"
        )

    def reset(self):
        self._running = None
        self.base_metric.reset()

    def update(self, target, pred):
        self.base_metric.update(target, pred)

    def compute(self):
        if self._running is None:
            self._running = self.base_metric.compute()
        else:
            self._running = (
                self._running * self.alpha
                + (1 - self.alpha) * self.base_metric.compute()
            )
        return self._running


class Reduce(Metric):
    def __init__(self, base_metric, reduction=lambda x: x):
        self.base_metric = base_metric
        self.reduction = reduction

    def reset(self):
        self.base_metric.reset()

    def update(self, target, pred):
        self.base_metric.update(target, pred)

    def compute(self):
        val = self.base_metric.compute()
        try:
            return self.reduction(val).item()  # tensor case
        except:
            return val  # scalar case


class Mean(Reduce):
    def __init__(self, base_metric):
        super().__init__(base_metric, reduction=torch.mean)


class Sum(Reduce):
    def __init__(self, base_metric):
        super().__init__(base_metric, reduction=torch.sum)


class Product(Reduce):
    def __init__(self, base_metric):
        super().__init__(base_metric, reduction=torch.prod)


class DistributedMetric(Metric, DataParallel):
    def __init__(self, metric, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__()  
        
    def reset(self):
        self.metric.reset()
        
    def update(self, target, pred):
        self.metric.update()
        
    def compute(self):
        self._sync_all()
        return self.metric.compute()
    
    def _sync_all(self):
        for attr in self._gather_variables:
            t = getattr(self, attr, None)
            if t is not None:
                t = self._sync_gather(t)
                setattr(self, attr, t)
        for attr in self._sum_variables:
            t = getattr(self, attr, None)
            if t is not None:
                t = self._sync_sum(t)
                setattr(self, attr, t)
                
    def _sync_sum(self):
        pass
    
    def _sync_all(self):
    
    
    
