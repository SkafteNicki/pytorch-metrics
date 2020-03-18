# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 12:10:16 2020

@author: nsde
"""

from pytorch_metrics import ClassificationMetric

class IoU(ClassificationMetric):
    name = 'iou'
    memory_efficient = True
    
    def reset(self):
        self._m11 = 0.0
        self._m00 = 0.0
        self._m01 = 0.0
        self._m10 = 0.0
        super().reset()
    
    def update(self, target, pred):
        self.check_input(target, pred)
        self.check_type(target, pred)
        target, pred = self.transform(target, pred)
        
        assert self._type == 'binary', 'Metric IoU can only be calculated for ' \
            'binary classification problems'
        
        self._m11 += (pred == 1 & target == 1).sum(dim=0)
        self._m00 += (pred == 0 & target == 0).sum(dim=0)
        self._m01 += (pred == 0 & target == 1).sum(dim=0)
        self._m10 += (pred == 1 & target == 0).sum(dim=0)
        
    def compute(self):
        val = self._m11 / (self._m00 + self._m01 + self._m10)
        try:
            return val.item()
        except:
            return val
        
        