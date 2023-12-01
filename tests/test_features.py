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

from pyheartlib.features import (
    get_hrv_features,
    get_stat_features,
    get_wf_feats,
)


@pytest.fixture
def data1():
    return np.array(
        [[9, 3, 2, 4, 1, 0, -11, -4], [-8, -4, 10, 7, 5, 2, 5, 1]],
        dtype=np.float32,
    )


@pytest.fixture
def rri_data():
    return np.array(
        [[300, 250, 330, 340, 200, 290], [230, 270, 310, 180, 220, 260]],
        dtype=np.float32,
    )


@pytest.fixture
def signal():
    return np.random.uniform(-1, 1, 600)


def test_get_stat_features(data1):
    flist = [
        "max",
        "min",
        "mean",
        "std",
        "median",
        "skew",
        "kurtosis",
        "range",
    ]
    feats_arr = get_stat_features(data1, features=flist)
    assert feats_arr.shape == (2, 8)
    assert feats_arr[0, 0] == 9.0
    assert feats_arr[0, 1] == -11.0
    assert feats_arr[1, 0] == 10.0
    assert feats_arr[1, 7] == 18.0


def test_get_hrv_features(rri_data):
    flist = [
        "meanrr",
        "sdrr",
        "medianrr",
        "rangerr",
        "nsdrr",
        "sdsd",
        "rmssd",
        "nrmssd",
    ]
    feats_arr = get_hrv_features(rri=rri_data, features=flist)
    assert feats_arr.shape == (2, 8)
    assert feats_arr[0, 0] == 285.0
    assert feats_arr[0, 1] == pytest.approx(47.87, rel=0.001)
    assert feats_arr[1, 0] == 245.0
    assert feats_arr[1, 7] == pytest.approx(0.2786, rel=0.001)


def test_get_wf_feats(signal):
    interval = 36
    feats = get_wf_feats(sig=signal, interval=interval, only_names=False)
    assert feats.shape == (16, 14)
    assert feats[0, 0] == np.max(signal[:interval])
    assert feats[0, 1] == np.min(signal[:interval])
