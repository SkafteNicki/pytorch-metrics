# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 17:13:37 2020

@author: nsde
"""

import torch
import numpy as np
import itertools
import pytorch_metrics as pm

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
)

import pytest
from testing_utils import can_run_gpu_test, move_to_positive

single_gpu = can_run_gpu_test()
TOL = 1e-2
N_SAMPLE = 100
N_UPDATE = 5
N_DIM = 10


metrics = [
    (pm.MeanSquaredError, mean_squared_error),
    (pm.MeanAbsoluteError, mean_absolute_error),
    (pm.ExplainedVariance, explained_variance_score),
]
devices = ["cpu", "cuda"]
test_list = metrics  # list(itertools.product(metrics, devices)) #TODO


def idfn(val):
    return str(val)


@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
def test_running_average(metric, baseline):
    alpha = 0.90
    n_repeats = 3
    TOL = 0.1  # TODO: make this more robust

    base_val = []
    m = pm.RunningAverage(metric(), alpha=alpha)
    for _ in range(n_repeats):
        target = np.random.randn(N_SAMPLE,)
        pred = np.random.randn(N_SAMPLE,)

        for i in range(N_UPDATE):  # do 10 updates
            m.update(torch.tensor(target[i::N_UPDATE]), torch.tensor(pred[i::N_UPDATE]))
        m_val = m.compute()

        base_val.append(baseline(target, pred))

    base_val = sum(
        [
            b * ((1 - alpha) ** (i != 0)) * (alpha ** (len(base_val) - (i + 1)))
            for i, b in enumerate(base_val)
        ]
    )
    assert abs(m_val - base_val) < TOL


def test_metric_collection():
    m1 = pm.MeanSquaredError()
    m2 = pm.MeanAbsoluteError()
    m3 = pm.ExplainedVariance()

    metrics = pm.MetricCollection([m1, m2, m3])

    target = np.random.randn(N_SAMPLE * N_UPDATE,)
    pred = np.random.randn(N_SAMPLE * N_UPDATE,)
    for i in range(N_UPDATE):
        input = (torch.tensor(target[i::N_UPDATE]), torch.tensor(pred[i::N_UPDATE]))
        m1.update(*input)
        m2.update(*input)
        m3.update(*input)
        metrics.update(*input)

    val1 = m1.compute()
    val2 = m2.compute()
    val3 = m3.compute()
    collection = metrics.compute()

    assert collection == {
        m1.name: val1,
        m2.name: val2,
        m3.name: val3,
    }, "MetricCollection not updated correctly"


test_list = [(pm.Sum, np.sum), (pm.Mean, np.mean), (pm.Product, np.product)]


@pytest.mark.parametrize("metric_reduction, baseline", test_list, ids=idfn)
def test_reductions(metric_reduction, baseline):

    target = np.random.randn(N_SAMPLE * N_UPDATE, N_DIM)
    pred = np.random.randn(N_SAMPLE * N_UPDATE, N_DIM)

    m = metric_reduction(pm.MeanSquaredError())
    for i in range(N_UPDATE):  # do 10 updates
        m.update(torch.tensor(target[i::N_UPDATE]), torch.tensor(pred[i::N_UPDATE]))
    m_val = m.compute()

    base_val = baseline(mean_squared_error(target, pred, multioutput="raw_values"))

    assert abs(m_val - base_val) < TOL
