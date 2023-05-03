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

Electrocardiogram (ECG) signals are the electrical activity of the heart as a graph of voltage versus time and have a significant importance in the healthcare. These signals contain valuable information regarding the heart health. Therefore they can be used for diagnosis and early detection of various cardiac conditions. Applying machine learning techniques on ECG signals can make their interperation faster and scaalebale.


# Statement of need

This paper introduces `pyecg`, a Python package for processing electrocardiogram signals. This package faciliates three necessary tasks for analysing ECG signals. The main functionality of this software is processing signals for heartbeat detection, heartbeat classification, and arrhythmia classification. Using this package researchers can focus on these tasks without the burden of designing data processing pipelines. The package supports both raw signals and computed features for model building. Therefore both traditional machine learning models and deep learning models can be trained for each task. The first release of this software works well with Keras and Tensorflow libraries and supports MIT-BIH Arrhythmia Database format. The package has optional preprocessing functionalities to remove noise and baseline wander from the raw signals.

# Example

An example model is provided along with this package which demonstrates a deep learning model designed using Keras library for heartbeat detection task. As this task is less complex compared to heartbeat and arhythmia classification tasks, it can be trained with high accuracy using the availble public datasets.
s
![Heartbeat detection example.\label{fig:example}](mis.png)

# References


