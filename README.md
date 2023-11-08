# pyheartlib

[![Documentation Status](https://readthedocs.org/projects/pyheartlib/badge/?version=latest)](https://pyheartlib.readthedocs.io/en/latest/?badge=latest)
![Workflow](https://github.com/devnums/pyheartlib/actions/workflows/ci-cd-main.yml/badge.svg?branch=main)
[![license-AGPL--3.0](https://img.shields.io/badge/license-AGPL--3.0-faa302)](https://github.com/devnums/pyheartlib/blob/main/LICENSE)
![Python Version](https://img.shields.io/pypi/pyversions/pyheartlib)
[![codecov](https://codecov.io/gh/devnums/pyheartlib/branch/main/graph/badge.svg?token=6IB18KL3E9)](https://codecov.io/gh/devnums/pyheartlib)
[![PyPI](https://img.shields.io/pypi/v/pyheartlib?color=blue)](https://pypi.org/project/pyheartlib/)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?logo=ubuntu&logoColor=white)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


Pyheartlib is a Python package for processing electrocardiogram recordings. This software facilitates working with signals for tasks such as heartbeat detection, heartbeat classification, and arrhythmia classification. Utilizing it, researchers can focus on these tasks without the burden of designing data processing modules. The package transforms original data into processed signal excerpts and their computed features in order to be used for training various machine learning models including advanced deep learning models, which can be trained by taking advantage of Keras and Tensorflow libraries.

## Documentation

Check out the [documentation](https://pyheartlib.readthedocs.io) for more information.

## Requirements

Current version of this package has been tested to work on the below spefication.

* Ubuntu:latest

* Python: 3.10 | 3.11

## Installation

The package can be installed with pip:

```bash
$ pip install pyheartlib
```

## Examples

Please check out the [example section](https://pyheartlib.readthedocs.io/en/latest/examples/examples.html) of the documentation.

Alternatively, you can access the [examples](https://github.com/devnums/pyheartlib/tree/main/examples) from the GitHub repository.

## License

AGPL-3.0
