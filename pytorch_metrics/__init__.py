"""
The library is a collection of the most used metrics within deep learning in
pytorch. It is intended to be easy to used with the API being simple and the
same for all the differnt metrics.

Each metric support two modes of evaluation: single evaluation and multi update
evaluation. In this single evaluation, we have a single target and prediction
and want to get the metric for this pair. We can get that value by calling:

.. code-block:: python

    m = Metric()
    val = m(target, prediction)
    
In the second case, we can imagine we have a loop over data, thus we do not
have access to all targets and predictions at the same time and we therefore
need to accumulate stats. In this case we can get the value by:
    
.. code-block:: python

    m = Metric()
    for data, target in Dataloader(...):
        prediction = model(data)
        m.update(target, prediction)
    val = m.compute()    

All metrics support cpu and gpu calculations (hopefully multi gpu and tpu
support in the future). If the documentation of the individual metrics are
missing, they should behave similar to their sklearn counterpart. 

"""

__version__ = "0.1"
__author__ = "Nicki Skafte Detlefsen"
__author_email__ = "nsde@dtu.dk"
__docs__ = (
    "Pytorch-metrics is a simple add on library to pytorch that adds "
    "many commonly used metrics within deep learning"
)

from .base import Metric, RegressionMetric, ClassificationMetric

from .wrappers import MetricCollection, RunningAverage, Mean, Sum, Product

from .regression import (
    MeanSquaredError,
    MeanAbsoluteError,
    RootMeanSquaredError,
    ExplainedVariance,
    R2Score,
    MaxError,
    MeanSquaredLogarithmicError,
    MeanTweedieDeviance,
    MeanPoissonDeviance,
    MeanGammaDeviance,
    CosineSimilarity,
    PearsonCorrelation,
    SpearmanCorrelation,
)

from .classification import (
    Accuracy,
    FilteredAccuracy,
    Precision,
    Recall,
    BalancedAccuracy,
    FBeta, F1,
    TopKAccuracy,
    ROC,
    AUC,
    ConfusionMatrix,
)

__all__ = [
    "Metric",
    "MetricDict",
    "RunningAverage",
    "MeanSquaredError",
    "MeanAbsoluteError",
    "RootMeanSquaredError",
    "ExplainedVariance",
    "R2Score",
    "MaxError",
    "MeanSquaredLogarithmicError",
    "MeanTweedieDeviance",
    "MeanPoissonDeviance",
    "MeanGammaDeviance",
    "CosineSimilary",
    "PearsonCorrelation",
    "SpearmanCorrelation",
    "Accuracy",
    "FilteredAccuracy",
    "Precision",
    "Recall",
    "BalancedAccuracy",
    "FBeta",
    "F1",
    "TopKAccuracy",
    "ROC",
    "AUC",
    "ConfusionMatrix"
]
