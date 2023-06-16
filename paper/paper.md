---
title: 'pyheartlib: A Python package for processing electrocardiogram signals'
tags:
  - Python
  - electrocardiogram
  - signal
  - heartbeat
  - arrhythmia
authors:
  - name: Sadegh Mohammadi
    orcid: 0000-0001-9763-4963
date: 19 June 2023
bibliography: paper.bib
---

# Summary

Electrocardiogram (ECG) signals represent the electrical activity of the heart as a graph of voltage versus time. These signals have significant importance in healthcare and contain valuable information. Therefore they can be analyzed for diagnosis and early detection of various cardiac conditions. 

# Statement of need

pyheartlib is a Python package for processing electrocardiogram signals. This software facilitates working with signals for tasks such as heartbeat detection, heartbeat classification, and arrhythmia classification. Using it, researchers can focus on these tasks without the burden of designing data processing modules. The package transforms original data into processed signal excerpts and their computed features which can be further utilized to train various machine learning models. Advanced deep learning models can be trained by taking advantage of Keras [@chollet2015keras] and Tensorflow [@tensorflow2015] libraries. The first release of this software supports WFDB format [@moody2001impact;@goldberger2000physiobank;@wfdb-python;@wfdb-software-package]. The package has optional preprocessing methods to remove noise and baseline wander from the raw signals.

# Example

As an example, a deep learning model was designed using Keras library for the heartbeat detection task. Because this task is less complex compared to the heartbeat and arrhythmia classification tasks, it can be trained with high accuracy using the available public datasets. 

![Example: heartbeat detection using deep learning.\label{fig:example}](mis.png){ width=80% }

# References