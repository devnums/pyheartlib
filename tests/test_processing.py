import pytest
from pyheartlib.processing import Processing
import numpy as np


@pytest.fixture
def signal():
    return np.random.uniform(-1, 1, 2000)


def test_remove_baseline(signal):
    r = Processing.remove_baseline(signal, sampling_rate=360)
    assert len(r) == len(signal)


def test_lowpass_filter_butter(signal):
    r = Processing.lowpass_filter_butter(
        signal, cutoff=45, sampling_rate=360, order=15
    )
    assert len(r) == len(signal)


def test_apply(signal):
    processors = [
        ("remove_baseline", dict(sampling_rate=360)),
        ("lowpass_filter_butter", dict(cutoff=45, sampling_rate=360, order=15)),
    ]
    r = Processing.apply(processors, signal)
    assert len(r) == len(signal)
