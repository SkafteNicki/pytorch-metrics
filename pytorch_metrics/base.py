# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:41:22 2020

@author: nsde
"""

from abc import ABC, abstractmethod

class Metric(ABC):
    """
    Base class
    """
    def __init__(self, 
                 transform = lambda x: x,
                 device = None):
        self.transform = transform
        
        # Initialize metric variables
        self.reset()
        
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def update(self, target, pred):
        pass
    
    @abstractmethod
    def compute(self):
        pass
    
    
        