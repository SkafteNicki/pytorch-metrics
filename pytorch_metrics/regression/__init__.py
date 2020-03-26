"""
Regression metrics are meant to be used in task where both the target and 
the predictions are real continues variables. The input to a regression metric

* `target`: either a `(N,)` vector with samples or a `(N,D)` vector with N samples
    from D different regression tasks.
* `pred`: either a `(N,)` vector with samples or a `(N,D)` vector with N samples
    from D different regression tasks. 

If `target` and `pred` is both `(N,)` vector of samples the output value will
be a single scalar, whereas if `target` and `pred` is matrices `(N,D)` the output
will be a `(D,)` torch tensor.
"""

from .meansquarederror import MeanSquaredError
from .meanabsoluterror import MeanAbsoluteError
from .rootmeansquarederror import RootMeanSquaredError
from .explainedvariance import ExplainedVariance
from .r2score import R2Score
from .maxerror import MaxError
from .meansquaredlogarithmicerror import MeanSquaredLogarithmicError
from .meantweediedeviance import MeanTweedieDeviance
from .meanpoissondeviance import MeanPoissonDeviance
from .meangammadeviance import MeanGammaDeviance
from .cosinesimilarity import CosineSimilarity
from .pearsoncorrelation import PearsonCorrelation
from .spearmancorrelation import SpearmanCorrelation
