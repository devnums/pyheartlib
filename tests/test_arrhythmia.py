import pytest
import numpy
import pandas
from pyecg.data_arrhythmia import ArrhythmiaData


@pytest.fixture
def arrhythmia_data():
    obj = ArrhythmiaData()
    return obj


@pytest.fixture
def record():
    r = {
        "signal": numpy.random.uniform(-1, 1, 230),
        "r_locations": [],
        "r_labels": [],
        "rhythms": ["(N", "(AFIB", "(T"],
        "rhythms_locations": [10, 100, 200],
    }
    return r


def test_full_annotate(arrhythmia_data, record):
    signal, full_ann = arrhythmia_data.full_annotate(record)
    assert len(signal) == len(full_ann)
    assert full_ann[0] == "unlab"
    assert full_ann[9] == "unlab"
    assert full_ann[10] == "(N"
    assert full_ann[99] == "(N"
    assert full_ann[100] == "(AFIB"
    assert full_ann[199] == "(AFIB"
    assert full_ann[200] == "(T"
    assert full_ann[-1] == "(T"
