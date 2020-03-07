<div align="center">
![Logo](docs/logo.svg)
</div>
---
Disclaimer: Work in progress.

The library is a collection o the most used metrics within deep learning in
pytorch. The library is hugely inspiret by [ignites.Metric](https://pytorch.org/ignite/metrics.html)
module, but is intented to be a standalon library for only computing metrics 
on pytorch tensor. 

Availble metrics:
* Regression metrics
    - MeanSquaredError
    - MeanAbsoluteError
    - RootMeanSquaredError
* Classification metrics
    - Accuracy
    - Recall 
    
## The API
When evaluating metrics, there are commonly two use cases:
* single evaluation
* multiple batch evaluation
pytorch-metric supports both uses. In the first cas the user can get the
wanted values by
```
m = Metric()
prediction = Model(data)
val = m(target, prediction)
```
in the second case, statistic needs to be accumulated during a iteration loop
```
m = Metric()
for data, target in DataLoader(   ):
    prediction = Model(data)
    m.update(target, prediction)
val = m.compute()
```


