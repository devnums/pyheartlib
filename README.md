# Pyheartlib

[![Documentation Status](https://readthedocs.org/projects/pyheartlib/badge/?version=latest)](https://pyheartlib.readthedocs.io/en/latest/?badge=latest)
![Workflow](https://github.com/devnums/pyheartlib/actions/workflows/ci-cd-main.yml/badge.svg?branch=main)
[![license-AGPL--3.0](https://img.shields.io/badge/license-AGPL--3.0-faa302)](https://github.com/devnums/pyheartlib/blob/main/LICENSE)
[![OS](https://img.shields.io/badge/os-Linux%20%7C%20macOS%20-yellow)](https://pyheartlib.readthedocs.io/en/latest/getting-started.html#requirements)
![Python Version](https://img.shields.io/pypi/pyversions/pyheartlib)
[![codecov](https://codecov.io/gh/devnums/pyheartlib/branch/main/graph/badge.svg?token=6IB18KL3E9)](https://codecov.io/gh/devnums/pyheartlib)
[![PyPI](https://img.shields.io/pypi/v/pyheartlib?color=blue)](https://pypi.org/project/pyheartlib/)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


`Pyheartlib` is a Python package for processing electrocardiogram (ECG) recordings. This software facilitates working with signals for tasks such as heartbeat detection, heartbeat classification, and arrhythmia classification. Utilizing it, researchers can focus on these tasks without the burden of designing data processing modules. The package transforms original data into processed signal excerpts and their computed features in order to be used for training various machine learning models including advanced deep learning models, which can be trained by taking advantage of Keras and Tensorflow libraries.

## Documentation

Documentation is available at the link below.

[pyheartlib.readthedocs.io](https://pyheartlib.readthedocs.io).

## Requirements

Current version of the package was tested on:

- Ubuntu: 20.04 | 22.04 & Python: 3.10 | 3.11 & Processor: x86_64

- macOS: 12.6.9 | 13.6 & Python: 3.10 | 3.11 & Processor: x86_64

However, it may also be compatible with other systems.

## Installation

The package can be installed with pip:

```bash
$ pip install pyheartlib
```

## Examples

Examples can be found in the [examples section](https://pyheartlib.readthedocs.io/en/latest/examples/examples.html) of the documentation and also in the GitHub repository ([examples](examples/)).

## Contributing

Feedback and contributions are appreciated. The guidelines for contributing are provided [here](CONTRIBUTING.md).

## Discussions & Support

For any questions, discussions, or problems with this software, please join us on [Discord](https://discord.gg/DQVfR2yWYc). An alternative option is to open a GitHub issue. ([Issues](https://github.com/devnums/pyheartlib/issues), [New issue](https://github.com/devnums/pyheartlib/issues/new))

## License

`Pyheartlib` is released under the [AGPL-3.0](LICENSE) License.
