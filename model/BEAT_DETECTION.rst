==============
Beat Detection
==============

This directory contains the code for training a deep learning model for the heartbeat detection task. As the task is not complex, it can be trained with high accuracy on few samples.

The ECG signals are segmented into 10 seconds ecxerpts. Each excerpt has a label of length 100 containing zeros and ones. For each 100ms segment the entry is one if it is a peak, otherwise it is zero.

The following steps prepares data and trains model.

Step 1: data_preparation.py
===========================

.. literalinclude:: data_preparation.py
  :language: python

Step 2: train.py
================

.. literalinclude:: train.py
  :language: python

Step 3: inference.py
====================

.. literalinclude:: inference.py
  :language: python


```bash
wget https://www.physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip
```


