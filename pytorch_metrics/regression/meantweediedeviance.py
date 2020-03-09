# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:34:10 2020

@author: nsde
"""

import torch
from pytorch_metrics import Metric
from pytorch_metrics.utils import check_non_zero_sample_size


def xlogy(x, y):
    z = x*y.log()
    z[x == 0] = 0
    return z


class MeanTweedieDeviance(Metric):
    name = 'meantweediedeviance'
    memory_efficient = True

    def __init__(self,
                 transform=lambda x, y: (x, y),
                 power=0):
        super().__init__(transform)
        self.power = power

    def reset(self):
        self._dev = 0
        self._n = 0

    def update(self, target, pred):
        message = (
            "Mean Tweedie deviance error with power={} can only be used on ".format(self.power))
        if self.power < 0:
            if torch.any(pred <= 0):
                raise ValueError(message + "strictly positive y_pred.")
            self._dev += 2 * (torch.pow(torch.max(target, 0), 2 - self.power)
                              / ((1 - self.power) * (2 - self.power))
                              - target *
                              torch.pow(pred, 1 - self.power)/(1 - self.power)
                              + torch.pow(pred, 2 - self.power)/(2 - self.power)).sum(dim=0)
        elif self.power == 0:
            self._dev += torch.pow(target - pred, 2.0).sum(dim=0)
        elif self.power < 1:
            raise ValueError("Tweedie deviance is only defined for power<=0 and "
                             "power>=1.")
        elif self.power == 1:
            if torch.any(target < 0) or torch.any(pred <= 0):
                raise ValueError(message + "non-negative y_true and strictly "
                                 "positive y_pred.")
            self._dev += 2 * (xlogy(target, target/pred) -
                              target + pred).sum(dim=0)
        elif self.power == 2:
            if (target <= 0).any() or (pred <= 0).any():
                raise ValueError(
                    message + "strictly positive y_true and y_pred.")
            self._dev += 2 * (torch.log(pred/target) +
                              target/pred - 1).sum(dim=0)
        else:
            if self.power < 2:
                if (target < 0).any() or (pred <= 0).any():
                    raise ValueError(message + "non-negative y_true and strictly "
                                     "positive y_pred.")
            else:
                if (target <= 0).any() or (pred <= 0).any():
                    raise ValueError(message + "strictly positive y_true and "
                                     "y_pred.")

            self._dev += 2 * (torch.pow(target, 2 - self.power)/((1 - self.power) * (2 - self.power))
                              - target *
                              torch.pow(pred, 1 - self.power)/(1 - self.power)
                              + torch.pow(pred, 2 - self.power)/(2 - self.power)).sum(dim=0)

        self._n += target.shape[0]

    def compute(self):
        check_non_zero_sample_size(self._n)
        val = self._dev / self._n
        try:
            return val.item()
        except:
            return val
