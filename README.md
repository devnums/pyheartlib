# Pyecg

`pyecg` is a Python package for processing and modeling electrocardiogram signals. This package facilitates processing signals for the task such as heartbeat detection, heartbeat classification, and arrhythmia classification. By using it, researchers can focus on these tasks without the burden of designing data processing modules. The package supports both raw signals and computed features for model building. Therefore traditional machine learning models and more advanced deep learning models can be used. The first release of this software supports Keras and Tensorflow libraries and MIT-BIH Arrhythmia Database format. The package has optional preprocessing methods to remove noise and baseline wander from the raw signals.

# Installation

To install `pyecg` with pip:

```bash
pip install pyecg
```

Check out the [documentation](https://pyecg.readthedocs.io) for more information and examples.


# Examples

Following examples demostrates dataset creation:

[Inter-patient dataset](examples/make_dataset_inter.py)
[Intra-patient dataset](examples/make_dataset_intra.py)
[Rpeak dataset](examples/make_dataset_rpeak.py)
[Arrhythmia dataset](examples/make_dataset_arrhythmia.py)


Following example demonstrates a deep learning model designed using Keras library for the heartbeat detection task. As this task is less complex compared to the heartbeat and arrhythmia classification tasks, it can be trained with high accuracy using the available public datasets.

[Beat detection model](model/BEAT_DETECTION.md)

