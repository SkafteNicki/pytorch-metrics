# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 07:54:43 2020

@author: nsde
"""

__version__ = '0.1'
__author__ = 'Nicki Skafte Detlefsen'
__author_email__ = 'nsde@dtu.dk'
__docs__ = 'Pytorch-metrics is a simple add on library to pytorch that adds '\
           'many commonly used metrics within deep learning'
           
from .regression import MeanSquaredError

__all__ = ['MeanSquaredError']