# pyheartlib

pyheartlib is a Python package for processing electrocardiogram signals. This software facilitates working with signals for tasks such as heartbeat detection, heartbeat classification, and arrhythmia classification. Using it, researchers can focus on these tasks without the burden of designing data processing modules. The package transforms original data into processed signal excerpts and their computed features which can be further utilized to train various machine learning models. Advanced deep learning models can be trained by taking advantage of Keras and Tensorflow libraries. The first release of this software supports WFDB format. The package has optional preprocessing methods to remove noise and baseline wander from the raw signals.

# Installation

The package can be installed with pip:

```bash
$ pip install pyheartlib
```

# Examples

Following examples demostrates dataset creation:

* {doc}`Inter-patient dataset <examples/dataset/inter_patient>` <br>
* {doc}`Intra-patient dataset <examples/dataset/intra_patient>` <br>
* {doc}`Rpeak dataset <examples/dataset/rpeak>` <br>
* {doc}`Arrhythmia dataset <examples/dataset/arrhythmia>`

As an example, a deep learning model is designed using the Keras library for the heartbeat detection task. Because this task is less complex compared to the heartbeat and arrhythmia classification tasks, it can be trained with high accuracy using available public datasets.

* {doc}`Beat detection model <examples/model/BEAT_DETECTION>`

![Example: heartbeat detection using deep learning.](examples/model/plots/mis.png)
