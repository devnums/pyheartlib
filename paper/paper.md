---
title: 'pyecg: A Python package for processing and modeling electrocardiogram signals'
tags:
  - Python
  - electrocardiogram
  - signal
  - heartbeat
  - arrhythmia
authors:
  - name: Sadegh Mohammadi
    orcid: 0000-0001-9763-4963
date: 4 May 2023
bibliography: paper.bib
---

# Summary

Electrocardiogram (ECG) signals represent the electrical activity of the heart as a graph of voltage versus time. These signals have significant importance in healthcare and contain valuable information. Therefore they can be analyzed for diagnosis and early detection of various cardiac conditions. 


# Statement of need

This paper introduces `pyecg`, a Python package for processing and modeling electrocardiogram signals. This package facilitates processing signals for the task such as heartbeat detection, heartbeat classification, and arrhythmia classification. By using it, researchers can focus on these tasks without the burden of designing data processing modules. The package supports both raw signals and computed features for model building. Therefore traditional machine learning models and more advanced deep learning models can be used. The first release of this software supports Keras [@chollet2015keras] and Tensorflow [@tensorflow2015] libraries and MIT-BIH Arrhythmia Database format [@moody2001impact;@goldberger2000physiobank]. The package has optional preprocessing methods to remove noise and baseline wander from the raw signals.

# Example

An example is provided along with this package which demonstrates a deep learning model designed using Keras library for the heartbeat detection task. As this task is less complex compared to the heartbeat and arrhythmia classification tasks, it can be trained with high accuracy using the available public datasets.

![Example: heartbeat detection.\label{fig:example}](mis.png){ width=80% }

# References