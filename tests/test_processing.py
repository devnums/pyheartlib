#############################################################################
# Copyright (c) 2023 Pyheartlib team. - All Rights Reserved                 #
# Project repo: https://github.com/devnums/pyheartlib                       #
# Contact: devnums.code@gmail.com                                           #
#                                                                           #
# This file is part of the Pyheartlib project.                              #
# To see the complete LICENSE file visit:                                   #
# https://github.com/devnums/pyheartlib/blob/main/LICENSE                   #
#############################################################################


import numpy as np
import pytest

from pyheartlib.processing import STFT, Processing


@pytest.fixture
def signal():
    return np.random.uniform(-1, 1, 2000)


def test_apply(signal):
    processors = [
        ("remove_baseline", dict(sampling_rate=360)),
        (
            "lowpass_filter_butter",
            dict(cutoff=45, sampling_rate=360, order=15),
        ),
    ]
    r = Processing.apply(processors, signal)
    assert len(r) == len(signal)


def test_custom_processors(signal):
    def processor1(x):
        return 2 * x

    def processor2(x):
        return x + 1

    custom_processors = [processor1, processor2]
    r = Processing.custom_processors(signal, processors=custom_processors)
    assert len(r) == len(signal)
    assert r[0] == pytest.approx(2 * signal[0] + 1, rel=0.001)
    assert r[-1] == pytest.approx(2 * signal[-1] + 1, rel=0.001)


def test_remove_baseline(signal):
    r = Processing.remove_baseline(signal, sampling_rate=360)
    assert len(r) == len(signal)


def test_lowpass_filter_butter(signal):
    r = Processing.lowpass_filter_butter(
        signal, cutoff=45, sampling_rate=360, order=15
    )
    assert len(r) == len(signal)


def test_denoise_signal(signal):
    r = Processing.denoise_signal(
        signal, remove_bl=True, lowpass=False, sampling_rate=360, order=15
    )
    assert len(r) == len(signal)


@pytest.fixture
def signal2d():
    return np.random.uniform(-1, 1, (3, 256))


def test_specgram(signal2d):
    dpr = STFT()
    features = dpr.specgram(
        signal2d, sampling_rate=360, nperseg=127, noverlap=122
    )
    assert features.shape == (3, 26, 64)


def test_calc_feat_dim():
    dpr = STFT()
    hdim, vdim = dpr.calc_feat_dim(samp=256, win=127, overlap=122)
    assert hdim == 26
    assert vdim == 64
    print(hdim, vdim)
