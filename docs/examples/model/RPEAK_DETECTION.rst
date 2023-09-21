================
R-peak Detection
================

This example contains the code for training a deep learning model for the R-peak detection task. As the task is not complex, it can be trained with high accuracy on a small amount of data.
In this example, the electrocardiogram signals are segmented into ten seconds ecxerpts. Each excerpt corresponds to an annotation list of length 600 containing zeros and ones. A subsegment with an R-peak corresponds to one in the annotation list.

The following steps prepare the data and train the model.


Data preparation
================

.. literalinclude:: data_preparation.py
  :language: python

Training
========

.. literalinclude:: train.py
  :language: python

Inference
=========

.. literalinclude:: inference.py
  :language: python

Results
=======

.. literalinclude:: result.txt
   :language: none

.. figure:: plots/mis.png
  :width: 600
  :alt: Example: R-peak detection using deep learning

  R-peak detection using deep learning.
