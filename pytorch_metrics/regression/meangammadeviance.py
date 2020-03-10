# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:55:45 2020

@author: nsde
"""

from .meantweediedeviance import MeanTweedieDeviance


class MeanGammaDeviance(MeanTweedieDeviance):
    name = 'meangammadeviance'
    memory_efficient = True

    def __init__(self, transform=None):
        super().__init__(transform, power=2)
