
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
    
Please see the [docs](https://pytorch-metrics.readthedocs.io/en/latest/) for more information.
