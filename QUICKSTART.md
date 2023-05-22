# Quickstart

To use this software, the first step is to place the raw data in a base data directory. Since the package supports MIT-BIH format, the MIT-BIH Arrhythmia Database can be used. It can be downloaded by runnig the following commands:

```
$ mkdir data
$ cd data
$ wget https://www.physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip
$ unzip mit-bih-arrhythmia-database-1.0.0.zip
```

A *config.yaml* file must be created in the base data directory. This file contains some required information about the data.

```
# Required fields:

DATA_DIR: "mit-bih-arrhythmia-database-1.0.0/"
BEAT_TYPES: ['N', 'L', 'R', 'j', 'e', 'V', 'E', 'A', 'S', 'a', 'J', 'F', 'f', '/', 'Q']
RHYTHM_TYPES: ['(AB', '(AFIB', '(AFL', '(B', '(BII', '(IVR', '(N', '(NOD', '(P', '(PREX', '(SBR', '(SVTA', '(T', '(VFL', '(VT']
```

The next step is to install the package with pip:

```bash
$ pip install pyheartlib
```

Check out the examples section of the [documentation](https://pyheartlib.readthedocs.io) for different use cases.
