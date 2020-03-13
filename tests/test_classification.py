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
    f1_score as _f1_score,
    roc_curve,
    roc_auc_score
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


def f1_score(y_true, y_pred):
    return _f1_score(y_true, y_pred, average="micro")


def filtered_accuracy(y_true, y_pred, ignore_label=[3, 7]):
    idx = 1 - np.array([y_true == l for l in ignore_label]).sum(axis=0)
    y_true = y_true[idx.astype("bool")]
    y_pred = y_pred[idx.astype("bool")]

    return accuracy_score(y_true, y_pred)

def topk_accuracy(y_true, y_pred, k=5):
    y_pred_sorted = np.argsort(y_pred, axis=-1)[...,::-1][...,:k]
    eq = y_true[...,np.newaxis] == y_pred_sorted
    return eq.sum(axis=-1).mean()


test_list = [
    (pm.Accuracy, accuracy_score, True),
    (pm.Precision, precision_score, True),
    (pm.Recall, recall_score, True),
    (partial(pm.FilteredAccuracy, labels_to_ignore=[3, 7]), filtered_accuracy, True),
    (pm.BalancedAccuracy, balanced_accuracy_score, True),
    (pm.F1, f1_score, True),
    (pm.TopKAccuracy, topk_accuracy, False)
]


def idfn(val):
    return str(val)


@pytest.mark.parametrize("metric, baseline, do_argmax", test_list, ids=idfn)
def test_single_update_cpu(metric, baseline, do_argmax):
    target = np.random.randint(0, N_CLASSES, N_SAMPLE)
    pred = softmax(np.random.randn(N_SAMPLE, N_CLASSES), axis=1)

    m = metric()
    m_val = m(torch.tensor(target), torch.tensor(pred))
    base_val = baseline(target, pred.argmax(axis=-1) if do_argmax else pred)

    assert abs(m_val - base_val) < TOL


@pytest.mark.parametrize("metric, baseline, do_argmax", test_list, ids=idfn)
def test_multi_update_cpu(metric, baseline, do_argmax):
    target = np.random.randint(0, N_CLASSES, N_SAMPLE * N_UPDATE)
    pred = softmax(np.random.randn(N_SAMPLE * N_UPDATE, N_CLASSES), axis=1)

    m = metric()
    for i in range(N_UPDATE):  # do 10 updates
        m.update(torch.tensor(target[i::N_UPDATE]), torch.tensor(pred[i::N_UPDATE]))
    m_val = m.compute()
    base_val = baseline(target, pred.argmax(axis=-1) if do_argmax else pred)

    assert abs(m_val - base_val) < TOL


@pytest.mark.skipif(not single_gpu, reason="Requires gpu")
@pytest.mark.parametrize("metric, baseline, do_argmax", test_list, ids=idfn)
def test_single_update_gpu(metric, baseline, do_argmax):
    target = np.random.randint(0, N_CLASSES, N_SAMPLE)
    pred = softmax(np.random.randn(N_SAMPLE, N_CLASSES), axis=1)

    m = metric()
    m_val = m(torch.tensor(target, device="cuda"), torch.tensor(pred, device="cuda"))
    base_val = baseline(target,pred.argmax(axis=-1) if do_argmax else pred)

    assert abs(m_val - base_val) < TOL


@pytest.mark.skipif(not single_gpu, reason="Requires gpu")
@pytest.mark.parametrize("metric, baseline, do_argmax", test_list, ids=idfn)
def test_multi_update_gpu(metric, baseline, do_argmax):
    target = np.random.randint(0, N_CLASSES, N_SAMPLE * N_UPDATE)
    pred = softmax(np.random.randn(N_SAMPLE * N_UPDATE, N_CLASSES), axis=1)

    m = metric()
    for i in range(N_UPDATE):  # do 10 updates
        m.update(
            torch.tensor(target[i::N_UPDATE], device="cuda"),
            torch.tensor(pred[i::N_UPDATE], device="cuda"),
        )
    m_val = m.compute()
    base_val = baseline(target, pred.argmax(axis=-1) if do_argmax else pred)

    assert abs(m_val - base_val) < TOL


def test_roc_and_auc_cpu():
    target = np.random.randint(0, 2, N_SAMPLE)
    pred = softmax(np.random.randn(N_SAMPLE, 2), axis=1)
    
    baseline_fpr, baseline_tpr, threshold = roc_curve(target, pred[:,1:])
    m = pm.ROC(line=torch.tensor(threshold))
    fpr, tpr = m(torch.tensor(target), torch.tensor(pred))
    #import pdb
    #pdb.set_trace()
    #print(baseline_fpr)
    #print(fpr)
    # for v1, v2 in zip(baseline_fpr[:-1], fpr[1:]):
    #     assert abs(v1 - v2.item()) < TOL
        
    # for v1, v2 in zip(baseline_tpr[:-1], tpr[1:]):
    #     assert abs(v1 - v2.item()) < TOL
        
    baseline_auc = roc_auc_score(target, pred[:,1:])
    m = pm.AUC()
    auc = m(torch.tensor(target), torch.tensor(pred))
    
    assert abs(baseline_auc - auc) < TOL