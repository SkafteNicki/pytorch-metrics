# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:55:45 2020

@author: nsde
"""

from .meantweediedeviance import MeanTweedieDeviance


class MeanPoissonDeviance(MeanTweedieDeviance):
    name = 'meanpoissondeviance'
    memory_efficient = True

    def __init__(self,
                 transform=lambda x, y: (x, y)):
        super().__init__(transform, power=1)
