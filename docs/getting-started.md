
# Getting Started


## Requirements

This version ({{ versionkey }}) of the package was tested on:

- Ubuntu: 20.04 | 22.04 & Python: 3.10 | 3.11

- macOS: 12.6.9 | 13.6 & Python: 3.10 | 3.11

## Installation

The package can be installed with pip:

```bash
$ pip install pyheartlib
```

## Data

`Pyheartlib` supports the *WFDB format*. Therefore, recordings in other formats must be converted before using this package.

The input data should be placed in a user-defined main data directory. This directory will hold the output datasets as well.

It is necessary to create a *config.yaml* file in the main data directory. This file contains some required fields regarding the original data, as described below.

:::{admonition} config.yaml
:class: hint

```bash
DATA_DIR: Directory of the input data relative to the main data directory.
SAMPLING_RATE: Sampling rate of the original signals.
CHANNEL: Name of the signal channel.
BEAT_TYPES: List of heartbeat types (R-peak labels).
RHYTHM_TYPES: List of rhythm types.
```
:::

:::{admonition} Example
:class: attention

To try this package on the [MIT-BIH Arrhythmia Database](https://www.physionet.org/content/mitdb/1.0.0/), the following commands can be executed in Linux to download the data in a main data directory named **data**.

```
$ mkdir data
$ cd data
$ wget https://www.physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip
$ unzip mit-bih-arrhythmia-database-1.0.0.zip
```
Below is the contents of the *config.yaml* file, which must be placed in the main data directory.

```
# file: data/config.yaml
DATA_DIR: "mit-bih-arrhythmia-database-1.0.0/"
SAMPLING_RATE: 360
CHANNEL: "MLII"
BEAT_TYPES: ['N', 'L', 'R', 'j', 'e', 'V', 'E', 'A', 'S', 'a', 'J', 'F', 'f', '/', 'Q']
RHYTHM_TYPES: ['(AB', '(AFIB', '(AFL', '(B', '(BII', '(IVR', '(N', '(NOD', '(P', '(PREX', '(SBR', '(SVTA', '(T', '(VFL', '(VT']
```
:::

## Usage

`Pyheartlib` contains three main classes for processing the original data and creating datasets.

### BeatData

For the heartbeat analysis task, the `BeatData` class can be used to make a dataset. To create a dataset using the `BeatData`, first it needs to be imported.
```python
from pyheartlib.data_beat import BeatData
```
The next step is to create an object of the `pyheartlib.data_beat.BeatData`.
```python
beatdata = BeatData(
    base_path="data",
    win=[200, 200],
    remove_bl=False,
    lowpass=False,
    progress_bar=False,
)
```
Descriptions of all the parameters of `BeatData` can be found [here](introduction.html#directive-bdt).
For feature computation, an instance of the `pyheartlib.beat_info.BeatInfo` has to be created.
```python
from pyheartlib.beat_info import BeatInfo
beatinfo = BeatInfo()
```
`BeatInfo` includes some predefined features, however user-defined features can also be added. Each user-defined feature definition must adhere to this syntax:
```python
def F_new_feature1(self):
    return return_result

# The `return_result` can be a single number or
# a dictionary such as {"Feature_1": value_1, "Feature_2": value_2}.
```

The user-defined features can be added to the beatinfo object by using the `add_features()` method. The list of all available features can be obtained using the `available_features()` method. To select desired features for computation the `select_features()` method can be used.
Descriptions of all the parameters of `BeatInfo` can be found [here](introduction.html#directive-bif).

Finally, using the `save_dataset_inter()` method of `BeatData`, an inter-patient dataset can be created.
```python
# The file will be saved in the base data directory.
beatdata.save_dataset_inter(["209", "215"], beatinfo, file="train.beat")
```

:::{admonition} Example
:class: attention

{doc}`Heartbeat dataset <examples/heartbeat>`
:::

### RhythmData

To create a dataset for arrhythmia classification, the `RhythmData` class can be used. This class can be imported using the code below.
```python
from pyheartlib.data_rhythm import RhythmData
```
The next step is to create an object of the `pyheartlib.data_rhythm.RhythmData`.
```python
rhythm_data = RhythmData(
    base_path="data", remove_bl=False, lowpass=False, progress_bar=False
)
```
Descriptions of all the parameters can be found [here](introduction.html#directive-rdt).

Using the `save_dataset()` method, the dataset will be created.
```python
rhythm_data.save_dataset(
    rec_list=train_set, file_name="train.arr", win_size=3600, stride=64
)
```
The dataset can be loaded using the `load_dataset()` function.
```python
from pyheartlib.data_rhythm import load_dataset
annotated_records, samples_info = load_dataset("data/train.arr")
```

 To generate batches of sample data, the dataset that was created before can be used. To accomplish this, an instance of `pyheartlib.data_rhythm.ECGSequence` must be created.
```python
from pyheartlib.data_rhythm import ECGSequence

trainseq = ECGSequence(
    annotated_records,
    samples_info,
    class_labels=None,
    batch_size=3,
    raw=True,
    interval=36,
    shuffle=False,
    rri_length=20
)
```

The `ECGSequence` takes the  *annotated_records* (ECG records) and the *samples_info* (metadata) that were loaded previously. Other parameters that `ECGSequence` takes can be found [here](introduction.html#directive-rsq).

:::{admonition} Example
:class: attention

{doc}`Arrhythmia dataset <examples/arrhythmia>`
:::

### RpeakData

To create a dataset using the `RpeakData`, first it needs to be imported.
```python
from pyheartlib.data_rpeak import RpeakData
```
The next step is to create an object of the `pyheartlib.data_rpeak.RpeakData`.
```python
rpeak_data = RpeakData(
    base_path="data", remove_bl=False, lowpass=False, progress_bar=False
)
```
Descriptions of all the parameters can be found [here](introduction.html#directive-rpd).

Using the `save_dataset()` method, the dataset will be created.

```python
rpeak_data.save_dataset(
    rec_list=train_set,
    file_name="train.rpeak",
    win_size=5 * 360,
    stride=360,
    interval=72,
)
```

The dataset can be loaded using the `load_dataset()` function.
```python
from pyheartlib.data_rpeak import load_dataset
annotated_records, samples_info = load_dataset("data/train.rpeak")
```
The dataset that was created can be used to generate batches of sample data. To accomplish this, an instance of `pyheartlib.data_rpeak.ECGSequence` must be created.
```python
from pyheartlib.data_rpeak import ECGSequence

trainseq = ECGSequence(
    annotated_records, samples_info, binary=False, batch_size=2, raw=True, interval=72
)
```

The `ECGSequence` takes the  *annotated_records* (ECG records) and the *samples_info* (metadata) that were loaded previously. Other parameters that `ECGSequence` takes can be found [here](introduction.html#directive-rps).

:::{admonition} Examples
:class: attention

- {doc}`R-peak dataset <examples/rpeak>`
- {doc}`R-peak detection <examples/rpeak_detection>`
:::
