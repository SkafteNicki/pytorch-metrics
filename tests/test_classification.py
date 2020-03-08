# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:38:09 2020

@author: nsde
"""

import math
import torch
import numpy as np
import pytorch_metrics as pm
from sklearn.metrics import (accuracy_score,
                             recall_score,
                             precision_score)
import pytest
from testing_utils import can_run_gpu_test

single_gpu = can_run_gpu_test()
TOL = 1e-2

test_list = [(pm.Accuracy, accuracy_score)]

def idfn(val):
    return str(val)

# @pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
# def test_single_update_cpu(metric, baseline):
#     pred=np.random.randint(0, 10, 100)
#     target=np.random.randint(0, 10, 100)
    
#     m = metric()
#     m_val = m(torch.tensor(pred), torch.tensor(target))
#     base_val = baseline(pred, target)

#     assert abs(m_val - base_val) < TOL
    
# @pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
# def test_multi_update_cpu(metric, baseline):
#     baseline_vals = [ ]
#     m = metric()
#     for _ in range(10): # do 10 updates
#         pred=np.random.randint(0, 10, 100)
#         target=np.random.randint(0, 10, 100)
#         m.update(torch.tensor(pred), torch.tensor(target))
        
#         baseline_vals.append(baseline(pred, target, normalize=False))
    
#     m_val = m.compute()
#     base_val = np.array(baseline_vals).sum() / 1000
    
#     assert abs(m_val - base_val) < TOL