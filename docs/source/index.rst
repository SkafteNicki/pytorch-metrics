.. pytorch-metrics documentation master file, created by
   sphinx-quickstart on Wed Mar 18 12:49:19 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pytorch-metrics's documentation!
===========================================

.. automodule:: pytorch_metrics

-------------------
Implemented metrics
-------------------

.. toctree::
   :maxdepth: 2
   :name: docs
   
   regression
   classification
   wrappers
   transforms

-------------------------
Note on memory efficiency
-------------------------

Most metrics can reduce their internal state when the user calls `update`, however
some metrics like `ExplainedVariance` requires access to the full set of targets
and predictions when `compute` is called. These metrics therefore store all
targets and predictions passed to `update` and are therefore not memory efficient.
Wheather or not a metric is memory efficient, is stored in the boolean variable
`Metric.memory_efficient`. If this is false, use the metric with care.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
