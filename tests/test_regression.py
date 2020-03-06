# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:15:29 2020

@author: nsde
"""

import torch
import numpy as np
import pytorch_metrics as pm
from sklearn.metrics import mean_squared_error

def test_metrics():
    pred=np.random.randn(100,)
    target=np.random.randn(100,)
    
    metric = pm.MeanSquaredError()
    metric.update(torch.tensor(pred), torch.tensor(target))
    pm_val = metric.compute()

    sk_val = mean_squared_error(pred, target)
    
    assert abs(pm_val.item() - sk_val) < 1e-4