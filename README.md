
![Logo](docs/logo.png)

---
The library is a collection of the most used metrics within deep learning in
pytorch. The library is hugely inspiret by [ignites.Metric](https://pytorch.org/ignite/metrics.html)
module, but is intented to be a standalon library for only computing metrics 
on pytorch tensor. The goal is to get as many of the [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) ported to pytorch.

Availble metrics:

* Regression metrics

    - MeanSquaredError
    - MeanAbsoluteError
    - RootMeanSquaredError
    - ExplainedVariance
    - R2Score
    - MaxError
    - MeanSquaredLogarithmicError
    - MeanTweedieDeviance
    - MeanPoissonDeviance
    - MeanGammaDeviance
    - CosineSimilarity
    - Correlation
    
* Classification metrics
    - Accuracy
    - BalancedAccuracy
    - FilteredAccuracy
    - Recall
    - Precision
    - BalancedAccuracy
    - F1
    - TopKAccuracy
    - ROC
    - AUC
    - ConfusionMatrix
    
The library have a number of wrappers that (as the name suggest) can be use in combination with the selection of metrics to offer extra functionality:
* MetricCollection
* RunningAverage
* Sum, Mean, Product 
    
## The API
When evaluating metrics, there are commonly two use cases:

* single evaluation
* multiple batch evaluation

pytorch-metric supports both uses. In the first case the user can get the wanted values by
```
m = Metric()
prediction = Model(data)
val = m(target, prediction)
```
in the second case, statistic needs to be accumulated during a iteration loop (i.e. running a training loop)
```
m = Metric()
for data, target in DataLoader(   ):
    prediction = Model(data)
    m.update(target, prediction)
val = m.compute()
```

# Memory efficiency

Most metrics can reduce their internal state when the user calls `update`, however
some metrics like `ExplainedVariance` requires access to the full set of targets
and predictions when `compute` is called. These metrics therefore store all
targets and predictions passed to `update` and are therefore not memory efficient.
Wheather or not a metric is memory efficient, is stored in the boolean variable
`Metric.memory_efficient`. If this is false, use the metric with care.