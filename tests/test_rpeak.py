import pytest
import numpy
import pandas
from pyecg.data_rpeak import RpeakData


@pytest.fixture
def rpeak_data():
    obj = RpeakData()
    return obj


@pytest.fixture
def record():
    r = {
        "signal": numpy.random.uniform(-1, 1, 1030),
        "r_locations": [200, 410, 650, 830],
        "r_labels": ["N", "V", "A", "A"],
        "rhythms": ["(N", "(AFIB", "(T"],
        "rhythms_locations": [10, 100, 200],
    }
    return r


def test_full_annotate(rpeak_data, record):
    signal, full_ann = rpeak_data.full_annotate(record)
    assert list(signal) == list(record["signal"])
    assert len(full_ann) == len(signal)
    assert full_ann[830] == "A"
    assert full_ann[200] == "N"
    assert full_ann[199] == 0
    assert full_ann[201] == 0
    assert len(set(full_ann)) == 4
    rlabels = record["r_labels"] + [0]
    assert list(set(full_ann)) == list(set(rlabels))
