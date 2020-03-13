# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:41:04 2020

@author: nsde
"""
import torch
from pytorch_metrics import ClassificationMetric
from pytorch_metrics.transforms import ROCTransform, DefaultTransform

class ROC(ClassificationMetric):
    name = 'roc'
    memory_efficient = False
    
    def __init__(self, transform=None, is_multilabel=False, line=None):
        super().__init__(transform, is_multilabel)
        if line is None:
            self.line = torch.linspace(0,1,100)
        else:
            self.line = line
            
    def reset(self):
        self._p = 0.0
        self._n = 0.0
        self._tp = 0.0
        self._fp = 0.0
        super().reset()

    def update(self, target, pred):
        self.check_input(target, pred)
        self.check_type(target, pred)
        target, pred = self.transform(target, pred)    
        
        self._p += (target==1).sum(dim=0)
        self._n += (target==0).sum(dim=0)
        self._tp += (target.unsqueeze(-1)*pred).sum(dim=0)
        self._fp += ((1 - target.unsqueeze(-1)) * pred).sum(dim=0)

    def compute(self):
        fpr = (self._fp / self._n).transpose(0,1).squeeze()
        tpr = (self._tp / self._p).transpose(0,1).squeeze()
        # TODO: add zero in front to make sure curve starts in (0,0)
        return fpr.squeeze(), tpr.squeeze() # squeeze if possible

    def check_type(self, target, pred):
        # do not change the default transform
        if pred.shape[-1] != 1 and pred.shape[-1] != 2:
            raise RuntimeError('ROC metric can only be used with binary '
                               'classification problems')
        
        _type = "binary"
        _num_classes = 2
        
        if isinstance(self.transform, DefaultTransform):
            self.transform = ROCTransform(self.line)
    
        if self._type is None:
            self._type = _type
            self._num_classes = _num_classes
        else:
            if self._type != _type:
                raise RuntimeError(
                    "Input data type has changed from {0} to {1}".format(
                        self._type, _type
                    )
                )
            if self._num_classes != _num_classes:
                raise RuntimeError(
                    "Input data has changed number of classes "
                    "from {0} to {1}".format(self._num_classes, _num_classes)
                )
                
class AUC(ROC):
    def compute(self):
        fpr, tpr = super().compute()
        # Use trapezoidal rule to calculate AUC
        auc = ((tpr[:-1] + tpr[1:]) / 2 * (fpr[:-1] - fpr[1:])).sum(dim=-1)
        try:
            return auc.item()
        except:
            return auc