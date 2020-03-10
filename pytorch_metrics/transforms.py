# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:48:38 2020

@author: nsde
"""

def compose_transform(*transforms):
    def f(target, pred):
        for t in transforms:
            target, pred = t(target, pred)
        return target, pred

def default_transform():
    def f(target, pred):
        return target, pred

def argmax_transform(dim=-1):
    def f(target, pred):
        return target, pred.argmax(dim=dim)
    return f

def round_transform(dim=-1):
    def f(target, pred):
        return target, pred.round(dim=dim)
    return f