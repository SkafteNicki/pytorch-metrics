# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:41:22 2020

@author: nsde
"""
import numbers
import torch
import warnings
from abc import ABC, abstractmethod
import torch.distributed as dist

class Metric(ABC):
    """
    Base class
    """

    def __init__(self,
                 transform=lambda x,y: (x,y),
                 device=None):
        self.transform = transform
        
        # Check device if distributed is initialized:
        if dist.is_available() and dist.is_initialized():

            # check if reset and update methods are decorated. Compute may not be decorated
            if not (hasattr(self.reset, "_decorated") and hasattr(self.update, "_decorated")):
                warnings.warn(
                    "{} class does not support distributed setting. Computed result is not collected "
                    "across all computing devices".format(self.__class__.__name__),
                    RuntimeWarning,
                )
            if device is None:
                device = "cuda"
            device = torch.device(device)
        self._device = device
        self._is_reduced = False
        
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
    
    def _sync_all_reduce(self, tensor):
        if not (dist.is_available() and dist.is_initialized()):
            # Nothing to reduce
            return tensor

        tensor_to_number = False
        if isinstance(tensor, numbers.Number):
            tensor = torch.tensor(tensor, device=self._device)
            tensor_to_number = True

        if isinstance(tensor, torch.Tensor):
            # check if the tensor is at specified device
            if tensor.device != self._device:
                tensor = tensor.to(self._device)
        else:
            raise TypeError("Unhandled input type {}".format(type(tensor)))

        # synchronize and reduce
        dist.barrier()
        dist.all_reduce(tensor)

        if tensor_to_number:
            return tensor.item()
        return tensor


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
                 alpha=0.98,
                 device=None):
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
