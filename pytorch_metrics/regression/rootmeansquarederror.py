# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:29:13 2020

@author: nsde
"""
import math
from .meansquarederror import MeanSquaredError

class RootMeanSquaredError(MeanSquaredError):
    
    def compute(self):
        mse = super().compute()
        return math.sqrt(mse)