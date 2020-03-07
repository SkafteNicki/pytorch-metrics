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
from testing_utils import can_run_gpu_test, can_run_multi_gpu_test

single_gpu = can_run_gpu_test()
multi_gpu = can_run_multi_gpu_test()
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
    m_val = m(torch.tensor(pred), torch.tensor(target))
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

@pytest.mark.skipif(not single_gpu, reason="Requires gpu")
@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)    
def test_single_update_gpu(metric, baseline):
    pred=np.random.randn(100,)
    target=np.random.randn(100,)
    
    m = metric(device='cuda')
    m_val = m(torch.tensor(pred, device='cuda'),
              torch.tensor(target, device='cuda'))

    base_val = baseline(pred, target)
    
    assert abs(m_val - base_val) < TOL    

@pytest.mark.skipif(not single_gpu, reason="Requires gpu")
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

@pytest.mark.skipif(not multi_gpu, reason="Requires multiple gpus")
@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
def test_single_update_ddp(metric, baseline):
    pred=np.random.randn(100,)
    target=np.random.randn(100,)
    
    m = metric(device='cuda')
    for device in ['cuda:0', 'cuda:1']:
        m.update(torch.tensor(pred[::2], device=device),
                 torch.tensor(target[::2], device=device))
    m_val = m.compute()    
    base_val = baseline(pred, target)
    
    assert abs(m_val - base_val) < TOL

@pytest.mark.skipif(not multi_gpu, reason="Requires multiple gpus")
@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
def test_multi_update_ddp(metric, baseline):
    baseline_vals = [ ]
    m = metric(device='cuda')
    for _ in range(10): # do 10 updates
        pred=np.random.randn(100,)
        target = np.random.randn(100,)
        for device in ['cuda:0', 'cuda:1']:
            m.update(torch.tensor(pred[::2], device='cuda'), 
                     torch.tensor(target[::2], device='cuda'))
        
        baseline_vals.append(baseline(pred, target))
    
    m_val = m.compute()
    base_val = np.array(baseline_vals).mean()
    
    assert abs(m_val - base_val) < TOL

    