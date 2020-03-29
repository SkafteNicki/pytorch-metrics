"""
Transforms are usefull for adjusting the target and predictions before the
metrics are calculated. For example in the classification case if the output of
a neural network is the class probabilities/logits i.e. tensor with shape (N,C),
the ArgmaxTransform(dim=-1) can be used to transform into labels before any
metrics are calculated.

To create your own transformation, you can simply inherent from the base class
`Transform` and implement your own `__init__` and `forward` methods. For an
example, here is the implementation of the ArgmaxTransform:
    
.. code-block:: python

    class ArgmaxTransform(Transform):
        def __init__(self, dim=-1):
            self.dim = dim

        def forward(self, target, pred):
            return target, pred.argmax(dim=self.dim)
        
The forward method of a transform is alwarys expected to take in two arguments
`(target, pred)` and return two variables, the transformed target and pred.

"""
from abc import ABC, abstractmethod


class Transform(ABC):
    def __init__(self):
        pass

    def __call__(self, target, pred):
        output = self.forward(target, pred)
        assert len(output)==2, "Transform should return two values: (target, pred)"
        return output

    @abstractmethod
    def forward(self, target, pred):
        pass


class ComposeTransforms(Transform):
    """ Compose multiple transforms into one single transform. The transforms
        will be compose in a sequential matter. 
    
    Args:
        transforms: sequence of transforms to compose into one.
    
    """
    def __init__(self, *transforms):
        self.transforms = transforms

    def forward(self, target, pred):
        for t in self.transforms:
            target, pred = t(target, pred)
        return target, pred

class DefaultTransform(Transform):
    """ Default transform that does nothing to the target and predictions """
    def forward(self, target, pred):
        return target, pred


class ArgmaxTransform(Transform):
    """ Applies the argmax transform to the predictions. Often used to transform
    probabilities/logits into labels in multiclass tasks.
    
    Args:
        dim: `int`, which dimension to apply the argmax transform over
    """
    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, target, pred):
        return target, pred.argmax(dim=self.dim)


class RoundTransform(Transform):
    """ Applies a round transform to the predictions. Often used to transform
    the output from sigmoid activations into labels in binary tasks 
    
    Args:
        dim: `int`, which dimension to apply the round transform over
    
    """
    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, target, pred):
        return target, pred.round(dim=self.dim)


class ROCTransform(Transform):
    def __init__(self, line):
        self.line = line

    def forward(self, target, pred):
        if pred.shape[-1] == 2:
            pred = pred[:, :, 1:]  # reduce to single dimension

        pred = (pred > self.line.type_as(pred)).long()
        return target, pred
