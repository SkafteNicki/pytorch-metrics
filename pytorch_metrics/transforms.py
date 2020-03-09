# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:48:38 2020

@author: nsde
"""

class MetricTransform:
    pass

class ArgmaxTransform(MetricTransform):
    pass

class SigmoidTransform(MetricTransform):
    pass

def argmax_transform(dim=-1):
    def f(target, pred):
        return target, pred.argmax(dim=dim)
    return f
    
def sigmoid_transform(dim=-1):
    def f(target, pred):
        return target, pred.round(dim=dim)
    return f
