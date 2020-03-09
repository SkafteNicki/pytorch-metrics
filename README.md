
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
* Classification metrics
    - Accuracy
    - (more comming soon)
    
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


