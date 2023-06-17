# pyheartlib

[![Documentation Status](https://readthedocs.org/projects/pyheartlib/badge/?version=latest)](https://pyheartlib.readthedocs.io/en/latest/?badge=latest)
![Workflow](https://github.com/devnums/pyheartlib/actions/workflows/ci-cd-main.yml/badge.svg?branch=main)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyheartlib)
[![codecov](https://codecov.io/gh/devnums/pyheartlib/branch/main/graph/badge.svg?token=6IB18KL3E9)](https://codecov.io/gh/devnums/pyheartlib)
![PyPI](https://img.shields.io/pypi/v/pyheartlib?color=blue)


pyheartlib is a Python package for processing electrocardiogram signals. This software facilitates working with signals for tasks such as heartbeat detection, heartbeat classification, and arrhythmia classification. Using it, researchers can focus on these tasks without the burden of designing data processing modules. The package transforms original data into processed signal excerpts and their computed features which can be further utilized to train various machine learning models. Advanced deep learning models can be trained by taking advantage of Keras and Tensorflow libraries. The first release of this software supports WFDB format. The package has optional preprocessing methods to remove noise and baseline wander from the raw signals.

# Installation

The package can be installed with pip:

```bash
$ pip install pyheartlib
```

# Dependencies

* python = ">=3.10,<3.12"
* numpy = ">=1.22.0"
* wfdb = ">=4.0.0"
* pandas = ">=1.4.0"
* tqdm = ">=4.63.0"
* scikit-learn = ">=1.1.0"
* tensorflow = ">=2.8.0"
* pyyaml = ">=6.0"
* matplotlib = ">=3.5.2"
* scipy = ">=1.8.0"

# Documentation

Check out the [documentation](https://pyheartlib.readthedocs.io) for more information.

# Examples

Following examples demostrate dataset creation:

* [Inter-patient dataset](https://github.com/devnums/pyheartlib/blob/main/examples/dataset/make_dataset_inter.py) <br>
* [Intra-patient dataset](https://github.com/devnums/pyheartlib/blob/main/examples/dataset/make_dataset_intra.py) <br>
* [Rpeak dataset](https://github.com/devnums/pyheartlib/blob/main/examples/dataset/make_dataset_rpeak.py) <br>
* [Arrhythmia dataset](https://github.com/devnums/pyheartlib/blob/main/examples/dataset/make_dataset_arrhythmia.py) <br>

As an example, a deep learning model is designed using the Keras library for the heartbeat detection task. Because this task is less complex compared to the heartbeat and arrhythmia classification tasks, it can be trained with high accuracy using available public datasets.

* [Beat detection model](https://github.com/devnums/pyheartlib/blob/main/examples/model/README.md)