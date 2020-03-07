# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:48:18 2020

@author: nsde
"""

from typing import Callable
from .base import Metric
from functools import wraps

def check_non_zero_sample_size(sample_size):
    if sample_size == 0:
        raise RuntimeError(
                'Must have one sample, before compute can be called'
                )

def sync_all_reduce(*attrs) -> Callable:
    def wrapper(func: Callable) -> Callable:
        @wraps(func)
        def another_wrapper(self: Metric, *args, **kwargs) -> Callable:
            if not isinstance(self, Metric):
                raise RuntimeError(
                    "Decorator sync_all_reduce should be used on " "ignite.metric.Metric class methods only"
                )

            if len(attrs) > 0 and not self._is_reduced:
                for attr in attrs:
                    t = getattr(self, attr, None)
                    if t is not None:
                        t = self._sync_all_reduce(t)
                        self._is_reduced = True
                        setattr(self, attr, t)

            return func(self, *args, **kwargs)

        return another_wrapper

    wrapper._decorated = True
    return wrapper


def set_is_reduced(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        self._is_reduced = False

    wrapper._decorated = True
    return wrapper