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
                             mean_absolute_error,
                             explained_variance_score,
                             r2_score,
                             max_error as _max_error)
import pytest
from testing_utils import can_run_gpu_test

single_gpu = can_run_gpu_test()
TOL = 1e-2
N_SAMPLE = 100
N_UPDATE = 5
N_DIM = 10

def root_mean_squared_error(y_true, y_pred, multioutput='uniform_average'):
    val = mean_squared_error(y_true, y_pred, multioutput=multioutput)
    try:
        return math.sqrt(val)
    except:
        return np.sqrt(val)

def max_error(y_true, y_pred, multioutput='uniform_average'):
    if multioutput == 'uniform_average':
        return _max_error(y_true, y_pred)
    else:
        return np.array([_max_error(yt, yp) for yt, yp in zip(y_true.T, y_pred.T)])

test_list = [(pm.MeanSquaredError, mean_squared_error),
             (pm.MeanAbsoluteError, mean_absolute_error),
             (pm.RootMeanSquaredError, root_mean_squared_error),
             (pm.ExplainedVariance, explained_variance_score),
             (pm.R2Score, r2_score),
             (pm.MaxError, max_error)]

def idfn(val):
    return str(val)

@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
def test_single_update_cpu(metric, baseline):
    target=np.random.randn(N_SAMPLE,)
    pred=np.random.randn(N_SAMPLE,)
    
    m = metric()
    m_val = m(torch.tensor(target), torch.tensor(pred))
    base_val = baseline(target, pred)
    
    assert abs(m_val - base_val) < TOL
    
@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
def test_multi_update_cpu(metric, baseline):
    target = np.random.randn(N_SAMPLE*N_UPDATE,)
    pred=np.random.randn(N_SAMPLE*N_UPDATE,)
    
    m = metric()
    for i in range(N_UPDATE): # do 10 updates
        m.update(torch.tensor(target[i::N_UPDATE]), torch.tensor(pred[i::N_UPDATE]))
    m_val = m.compute()
    base_val = baseline(target, pred)
    
    assert abs(m_val - base_val) < TOL

@pytest.mark.skipif(not single_gpu, reason="Requires gpu")
@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)    
def test_single_update_gpu(metric, baseline):
    target=np.random.randn(N_SAMPLE,)
    pred=np.random.randn(N_SAMPLE,)
    
    m = metric()
    m_val = m(torch.tensor(target, device='cuda'),
              torch.tensor(pred, device='cuda'))

    base_val = baseline(target, pred)
    
    assert abs(m_val - base_val) < TOL    

@pytest.mark.skipif(not single_gpu, reason="Requires gpu")
@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
def test_multi_update_gpu(metric, baseline):
    target = np.random.randn(N_SAMPLE*N_UPDATE,)
    pred=np.random.randn(N_SAMPLE*N_UPDATE,)
    
    m = metric()
    for i in range(N_UPDATE): # do 10 updates
        m.update(torch.tensor(target[i::N_UPDATE], device='cuda'), 
                 torch.tensor(pred[i::N_UPDATE], device='cuda'))    
    m_val = m.compute()
    
    base_val = baseline(target, pred)
    
    assert abs(m_val - base_val) < TOL
    
@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
def test_single_update_cpu_batch(metric, baseline):
    target=np.random.randn(N_SAMPLE,N_DIM)
    pred=np.random.randn(N_SAMPLE,N_DIM)
    
    m = metric()
    m_val = m(torch.tensor(pred), torch.tensor(target))
    base_val = baseline(pred, target, multioutput='raw_values')

    for v,b in zip(m_val,base_val):
        assert abs(v.item() - b) < TOL
    
@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
def test_multi_update_cpu_batch(metric, baseline):
    target = np.random.randn(N_SAMPLE*N_UPDATE,N_DIM)
    pred=np.random.randn(N_SAMPLE*N_UPDATE,N_DIM)
    
    m = metric()
    for i in range(N_UPDATE): # do 10 updates
        m.update(torch.tensor(target[i::N_UPDATE]), torch.tensor(pred[i::N_UPDATE]))
    m_val = m.compute()
    
    base_val = baseline(target, pred, multioutput='raw_values')
  
    for v, b in zip(m_val,base_val):
        assert abs(v.item() - b) < TOL

@pytest.mark.skipif(not single_gpu, reason="Requires gpu")        
@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
def test_single_update_gpu_batch(metric, baseline):
    target=np.random.randn(N_SAMPLE,N_DIM)
    pred=np.random.randn(N_SAMPLE,N_DIM)
    
    m = metric()
    m_val = m(torch.tensor(pred, device='cuda'), 
              torch.tensor(target, device='cuda'))
    base_val = baseline(pred, target, multioutput='raw_values')

    for v,b in zip(m_val,base_val):
        assert abs(v.item() - b) < TOL

@pytest.mark.skipif(not single_gpu, reason="Requires gpu")    
@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
def test_multi_update_gpu_batch(metric, baseline):
    target = np.random.randn(N_SAMPLE*N_UPDATE,N_DIM)
    pred=np.random.randn(N_SAMPLE*N_UPDATE,N_DIM)
    
    m = metric()
    for i in range(N_UPDATE): # do 10 updates
        m.update(torch.tensor(target[i::N_UPDATE], device='cuda'), 
                 torch.tensor(pred[i::N_UPDATE], device='cuda'))
    m_val = m.compute()
    
    base_val = baseline(target, pred, multioutput='raw_values')
  
    for v, b in zip(m_val,base_val):
        assert abs(v.item() - b) < TOL