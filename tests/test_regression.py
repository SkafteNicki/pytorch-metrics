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

TOL = 1e-2

def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

test_list = [(pm.MeanSquaredError, mean_squared_error),
             (pm.MeanAbsoluteError, mean_absolute_error),
             (pm.RootMeanSquaredError, root_mean_squared_error)]

def idfn(val):
    return str(val)

@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
def test_single_update_cpu(metric, baseline):
    pred=np.random.randn(100,)
    target=np.random.randn(100,)
    
    m = metric()
    m.update(torch.tensor(pred), torch.tensor(target))
    m_val = m.compute()

    base_val = baseline(pred, target)
    
    assert abs(m_val - base_val) < TOL
    
@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
def test_multi_update_cpu(metric, baseline):
    baseline_vals = [ ]
    m = metric()
    for _ in range(10): # do 10 updates
        pred=np.random.randn(100,)
        target = np.random.randn(100,)
        m.update(torch.tensor(pred), torch.tensor(target))
        
        baseline_vals.append(baseline(pred, target))
    
    m_val = m.compute()
    base_val = np.array(baseline_vals).mean()
    
    assert abs(m_val - base_val) < TOL

@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)    
def test_single_update_gpu(metric, baseline):
    pred=np.random.randn(100,)
    target=np.random.randn(100,)
    
    m = metric(device='cuda')
    m.update(torch.tensor(pred, device='cuda'),
             torch.tensor(target, device='cuda'))
    m_val = m.compute()

    base_val = baseline(pred, target)
    
    assert abs(m_val - base_val) < TOL    

@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
def test_multi_update_gpu(metric, baseline):
    baseline_vals = [ ]
    m = metric(device='cuda')
    for _ in range(10): # do 10 updates
        pred=np.random.randn(100,)
        target = np.random.randn(100,)
        m.update(torch.tensor(pred, device='cuda'), 
                 torch.tensor(target, device='cuda'))
        
        baseline_vals.append(baseline(pred, target))
    
    m_val = m.compute()
    base_val = np.array(baseline_vals).mean()
    
    assert abs(m_val - base_val) < TOL