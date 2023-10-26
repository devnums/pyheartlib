================
R-peak Detection
================

This example contains the code for training a deep learning model for the R-peak detection task. As the task is not complex, it can be trained with high accuracy on a small amount of data.
In this example, the electrocardiogram signals are segmented into ten seconds ecxerpts. Each excerpt corresponds to an annotation list of length 600 containing zeros and ones. A subsegment with an R-peak corresponds to one in the annotation list.

This example is available on `GitHub <https://github.com/devnums/pyheartlib/blob/main/examples/>`_.

.. raw:: html

    <a target="_blank" href="https://colab.research.google.com/github/devnums/pyheartlib/blob/main/examples/model/rpeak_detection.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


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
