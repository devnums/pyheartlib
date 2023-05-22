# pyheartlib

pyheartlib is a Python package for processing and modeling electrocardiogram signals. This package facilitates processing signals for the task such as heartbeat detection, heartbeat classification, and arrhythmia classification. By using it, researchers can focus on these tasks without the burden of designing data processing modules. The package supports both raw signals and computed features for model building. Therefore traditional machine learning models and more advanced deep learning models can be used. The first release of this software supports Keras and Tensorflow libraries, and MIT-BIH Arrhythmia Database format. The package has optional preprocessing methods to remove noise and baseline wander from the raw signals.

# Installation

The package can be installed with pip:

```bash
$ pip install pyheartlib
```

# Requirements

pyheartlib has been tested on Ubuntu, Python 3.11 and the depends on the packages.

* numpy>=1.21.1
* pandas>=1.3.1

# Documentation

Check out the [documentation](https://pyheartlib.readthedocs.io) for more information.

# Examples

Following examples demostrate dataset creation:

* [Inter-patient dataset](examples/make_dataset_inter.py) <br>
* [Intra-patient dataset](examples/make_dataset_intra.py) <br>
* [Rpeak dataset](examples/make_dataset_rpeak.py) <br>
* [Arrhythmia dataset](examples/make_dataset_arrhythmia.py) <br>

As an example, a deep learning model was designed using Keras library for the heartbeat detection task. Because this task is less complex compared to the heartbeat and arrhythmia classification tasks, it can be trained with high accuracy using the available public datasets.

* [Beat detection model](model/README.md)