# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:48:18 2020

@author: nsde
"""

def check_non_zero_sample_size(sample_size):
    if sample_size == 0:
        raise RuntimeError(
                'Must have one sample, before compute can be called'
                )

def atleast_2d(tensor):
    if len(tensor.shape)<2:
        return tensor.unsqueeze(1)
    else:
        return tensor
    