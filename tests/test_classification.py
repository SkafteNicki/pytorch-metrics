# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:38:09 2020

@author: nsde
"""

import math
import torch
import numpy as np
from functools import partial
import pytorch_metrics as pm
from sklearn.metrics import (
    accuracy_score,
    recall_score as _recall_score,
    precision_score as _precision_score,
    balanced_accuracy_score,
)
from scipy.special import softmax
import pytest
from testing_utils import can_run_gpu_test

single_gpu = can_run_gpu_test()
TOL = 1e-2
N_SAMPLE = 100
N_UPDATE = 5
N_DIM = 10
N_CLASSES = 10


def precision_score(y_true, y_pred):
    return _precision_score(y_true, y_pred, average="micro")


def recall_score(y_true, y_pred):
    return _recall_score(y_true, y_pred, average="micro")


def filtered_accuracy(y_true, y_pred, ignore_label=[3, 7]):
    idx = 1 - np.array([y_true == l for l in ignore_label]).sum(axis=0)
    y_true = y_true[idx.astype("bool")]
    y_pred = y_pred[idx.astype("bool")]

    return accuracy_score(y_true, y_pred)


test_list = [
    (pm.Accuracy, accuracy_score),
    (pm.Precision, precision_score),
    (pm.Recall, recall_score),
    (partial(pm.FilteredAccuracy, labels_to_ignore=[3, 7]), filtered_accuracy),
    (pm.BalancedAccuracy, balanced_accuracy_score),
]


def idfn(val):
    return str(val)


@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
def test_single_update_cpu(metric, baseline):
    target = np.random.randint(0, N_CLASSES, N_SAMPLE)
    pred = softmax(np.random.randn(N_SAMPLE, N_CLASSES), axis=1)

    m = metric()
    m_val = m(torch.tensor(target), torch.tensor(pred))
    base_val = baseline(target, pred.argmax(axis=-1))

    assert abs(m_val - base_val) < TOL


@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
def test_multi_update_cpu(metric, baseline):
    target = np.random.randint(0, N_CLASSES, N_SAMPLE * N_UPDATE)
    pred = softmax(np.random.randn(N_SAMPLE * N_UPDATE, N_CLASSES), axis=1)

    m = metric()
    for i in range(N_UPDATE):  # do 10 updates
        m.update(torch.tensor(target[i::N_UPDATE]), torch.tensor(pred[i::N_UPDATE]))
    m_val = m.compute()
    base_val = baseline(target, pred.argmax(axis=-1))

    assert abs(m_val - base_val) < TOL


@pytest.mark.skipif(not single_gpu, reason="Requires gpu")
@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
def test_single_update_gpu(metric, baseline):
    target = np.random.randint(0, N_CLASSES, N_SAMPLE)
    pred = softmax(np.random.randn(N_SAMPLE, N_CLASSES), axis=1)

    m = metric()
    m_val = m(torch.tensor(target, device="cuda"), torch.tensor(pred, device="cuda"))
    base_val = baseline(target, pred.argmax(axis=-1))

    assert abs(m_val - base_val) < TOL


@pytest.mark.skipif(not single_gpu, reason="Requires gpu")
@pytest.mark.parametrize("metric, baseline", test_list, ids=idfn)
def test_multi_update_gpu(metric, baseline):
    target = np.random.randint(0, N_CLASSES, N_SAMPLE * N_UPDATE)
    pred = softmax(np.random.randn(N_SAMPLE * N_UPDATE, N_CLASSES), axis=1)

    m = metric()
    for i in range(N_UPDATE):  # do 10 updates
        m.update(
            torch.tensor(target[i::N_UPDATE], device="cuda"),
            torch.tensor(pred[i::N_UPDATE], device="cuda"),
        )
    m_val = m.compute()
    base_val = baseline(target, pred.argmax(axis=-1))

    assert abs(m_val - base_val) < TOL
