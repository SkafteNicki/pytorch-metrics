# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:14:03 2020

@author: nsde
"""
import warnings
import torch
import numpy as np


def can_run_gpu_test():
    if torch.cuda.is_available():
        return True
    else:
        warnings.warn('Cannot run gpu tests')
        return False


def can_run_multi_gpu_test():
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return True
    else:
        warnings.warn('Cannot run multi gpu tests')
        return False


def move_to_positive(target, pred):
    target += np.abs(target.min())*1.01
    pred += np.abs(pred.min())*1.01
    return target, pred
