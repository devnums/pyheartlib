# pyheartlib

pyheartlib is a Python package for processing and modeling electrocardiogram signals. This package facilitates processing signals for the task such as heartbeat detection, heartbeat classification, and arrhythmia classification. By using it, researchers can focus on these tasks without the burden of designing data processing modules. The package supports both raw signals and computed features for model building. Therefore traditional machine learning models and more advanced deep learning models can be used. The first release of this software supports Keras and Tensorflow libraries, and MIT-BIH Arrhythmia Database format. The package has optional preprocessing methods to remove noise and baseline wander from the raw signals.

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

As an example, a deep learning model was designed using Keras library for the heartbeat detection task. Because this task is less complex compared to the heartbeat and arrhythmia classification tasks, it can be trained with high accuracy using the available public datasets.

* {doc}`Beat detection model <examples/model/BEAT_DETECTION>`

![Example: heartbeat detection using deep learning.](examples/model/plots/mis.png)