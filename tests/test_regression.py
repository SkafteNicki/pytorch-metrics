# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:15:29 2020

@author: nsde
"""
import math
import torch
import numpy as np
import pytorch_metrics as pm
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error)
import pytest

def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

test_list = [(pm.MeanSquaredError, mean_squared_error),
             (pm.MeanAbsoluteError, mean_absolute_error),
             (pm.RootMeanSquaredError, root_mean_squared_error)]

def idfn(val):
    return str(val)

@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
def test_metrics(metric, baseline):
    pred=np.random.randn(100,)
    target=np.random.randn(100,)
    
    m = metric()
    m.update(torch.tensor(pred), torch.tensor(target))
    m_val = m.compute()

    base_val = baseline(pred, target)
    
    assert abs(m_val - base_val) < 1e-4