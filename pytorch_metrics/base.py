# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:41:22 2020

@author: nsde
"""

from abc import ABC, abstractmethod
from .transforms import DefaultTransform, ArgmaxTransform, RoundTransform


class Metric(ABC):
    """
    Base class
    """

    name = "metric"
    memory_efficient = True

    def __init__(self, transform=None):
        if transform is None:
            self.transform = DefaultTransform()
        else:
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


class RegressionMetric(Metric):
    name = "regressionmetric"
    memory_efficient = True

    def check_input(self, target, pred):
        ts = target.shape
        ps = pred.shape

        assert ts == ps, "Target and pred needs to have same shape"

        if len(ts) < 1:
            target.unsqueeze_(0)

        if len(ts) < 2:
            target.unsqueeze_(1)

        if len(ps) < 1:
            pred.unsqueeze_(0)

        if len(ps) < 2:
            pred.unsqueeze_(1)


class ClassificationMetric(Metric):
    name = "classificationmetric"
    memory_efficient = True

    def __init__(self, transform=None, is_multilabel=False):
        super().__init__(transform)
        self._is_multilabel = is_multilabel
        self._type = None
        self._num_classes = None

    def reset(self):
        self._type = None
        self._num_classes = None

    def check_input(self, target, pred):
        ts = target.shape
        ps = pred.shape

        assert (
            target.squeeze().shape == pred.squeeze().shape[:-1]
        ), "Target can only differ in the last dimension"

        # Unsqueeze such that target.shape = [N,d] and
        # pred.shape = [N,d,C]
        if len(ts) < 0:
            target.unsqueeze_(0)

        if len(ts) < 1:
            target.unsqueeze_(0)

        if len(ts) < 2:
            target.unsqueeze_(1)

        if len(ps) < 0:
            pred.unsqueeze_(0)

        if len(ps) < 1:
            pred.unsqueeze_(0)

        if len(ps) < 2:
            pred.unsqueeze_(0)

        if len(ps) < 3:
            pred.unsqueeze_(1)

    def check_type(self, target, pred):
        if pred.shape[-1] == 1:
            _type = "binary"
            if isinstance(self.transform, DefaultTransform):
                self.transform = RoundTransform()
            _num_classes = 2
        elif pred.shape[-1] == 2:
            _type = "binary"
            if isinstance(self.transform, DefaultTransform):
                self.transform = ArgmaxTransform()
            _num_classes = 2
        else:
            _type = "multiclass"
            if isinstance(self.transform, DefaultTransform):
                self.transform = ArgmaxTransform()
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
