# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 17:13:37 2020

@author: nsde
"""

import torch
import numpy as np
import itertools
import pytorch_metrics as pm

from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             explained_variance_score)

import pytest
from testing_utils import (can_run_gpu_test,
                           move_to_positive)

single_gpu = can_run_gpu_test()
TOL = 1e-2
N_SAMPLE = 100
N_UPDATE = 5
N_DIM = 10


metrics = [(pm.MeanSquaredError, mean_squared_error),
           (pm.MeanAbsoluteError, mean_absolute_error),
           (pm.ExplainedVariance, explained_variance_score)]
devices = ['cpu', 'cuda']
test_list = metrics  # list(itertools.product(metrics, devices)) #TODO


def idfn(val):
    return str(val)

# @pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
# def test_running_average(metric, baseline):
#     alpha = 0.99
#     n_repeats = 2

#     base_val=[]
#     m = pm.RunningAverage(metric(), alpha=alpha)
#     for _ in range(n_repeats):
#         target=np.random.randn(N_SAMPLE,)
#         pred=np.random.randn(N_SAMPLE,)

#         for i in range(N_UPDATE): # do 10 updates
#             m.update(torch.tensor(target[i::N_UPDATE]), torch.tensor(pred[i::N_UPDATE]))
#         m_val = m.compute()

#         base_val.append(baseline(target, pred))

#     base_val = sum([b*(alpha)**i for i,b in enumerate(base_val)])

#     assert abs(m_val - base_val) < TOL

# def test_metric_dict1:
#     metrics = pm.MetricDict([
#         pm.MeanSquaredError(),
#         pm.MeanAbsoluteError(),
#         pm.ExplainedVariance()
#         ])


# def test_metric_dict2:
#     pass
