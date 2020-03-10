# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:39:09 2020

@author: nsde
"""
from pytorch_metrics import Metric


class ConfusionMatrix(Metric):
    def __init__(self, num_classes, transform=lambda x, y: (x, y), device=None):
        super().__init__(transform, device)
