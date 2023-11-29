#############################################################################
# Copyright (c) 2023 Pyheartlib team. - All Rights Reserved                 #
# Project repo: https://github.com/devnums/pyheartlib                       #
# Contact: devnums.code@gmail.com                                           #
#                                                                           #
# This file is part of the Pyheartlib project.                              #
# To see the complete LICENSE file visit:                                   #
# https://github.com/devnums/pyheartlib/blob/main/LICENSE                   #
#############################################################################


import numpy
import pytest

from pyheartlib.beat_info import BeatInfo


@pytest.fixture
def test_data():
    dic = {
        "waveform": numpy.random.uniform(low=-1, high=1, size=(300, 2)),
        "rpeak_locs": [30, 115, 180, 230],  # beat at index 2
        "rec_id": 100,
        "start_idx": 69,
        "label": "N",
    }
    return dic


@pytest.fixture
def beatinfo():
    obj = BeatInfo(beat_loc=2)
    return obj


def test_available_features(beatinfo):
    assert isinstance(beatinfo.available_features(), list)


def test_add_features(test_data, beatinfo):
    def F_new_feature1(self):
        return 890

    def F_new_feature2(self):
        return {"FEAT_a": 330, "FEAT_b": 4668}

    def F_new_feature2b(self):
        return {"FEAT_a2": (23, 67), "FEAT_b2": (35, 76)}

    def F_new_feature2c(self):
        return {
            "FEAT_a3": numpy.array([89.3, 58]),
            "FEAT_b3": numpy.array([35.3, 47.5]),
        }

    def F_new_feature3(self):
        return (218, 4.5)

    def F_new_feature3b(self):
        return numpy.array([56.2, 236])

    def F_new_feature4(self):
        return [3, 4, 2, 6, 27]

    new_features = [
        F_new_feature1,
        F_new_feature2,
        F_new_feature2b,
        F_new_feature2c,
        F_new_feature3,
        F_new_feature3b,
        F_new_feature4,
    ]
    beatinfo.add_features(new_features)
    assert hasattr(beatinfo, "F_new_feature1")
    assert hasattr(beatinfo, "F_new_feature2")
    assert "F_new_feature1" in beatinfo.available_features()

    beatinfo(test_data)
    assert "F_new_feature1" in beatinfo.features.keys()
    assert beatinfo.features["F_new_feature1"] == 890
    assert beatinfo.features["FEAT_a"] == 330
    assert beatinfo.features["FEAT_a2"] == (23, 67)
    assert beatinfo.features["FEAT_b3"] == (35.3, 47.5)
    assert beatinfo.features["F_new_feature3(CH2)"] == 4.5
    assert beatinfo.features["F_new_feature3b(CH1)"] == 56.2
    assert beatinfo.features["F_new_feature4"] == [3, 4, 2, 6, 27]


def test_select_features(beatinfo):
    feats = ["feature1", "feature2"]
    beatinfo.select_features(feats)
    assert set(beatinfo.selected_features_names) == set(feats)
