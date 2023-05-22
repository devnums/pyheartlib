import pytest
import numpy
import pandas
from pyheartlib.beat_info import BeatInfo


signal_frag = numpy.random.uniform(-1, 1, 300).tolist()
beat_type = "N"
rec_id = 100
r_locs = [30, 115, 180, 230]  # beat at index 2
s_idx = 69


@pytest.fixture
def beatinfo():
    obj = BeatInfo(beat_loc=2)
    return obj


def test_available_features(beatinfo):
    assert isinstance(beatinfo.available_features(), list)


def test_add_features(beatinfo):
    def F_new_feature1(self):
        return 890

    def F_new_feature2(self):
        return 330

    new_features = [F_new_feature1, F_new_feature2]
    beatinfo.add_features(new_features)
    assert hasattr(beatinfo, "F_new_feature1")
    assert hasattr(beatinfo, "F_new_feature2")
    assert "F_new_feature1" in beatinfo.available_features()

    beatinfo(
        {
            "waveform": signal_frag,
            "rpeak_locs": r_locs,
            "rec_id": rec_id,
            "start_idx": s_idx,
            "label": beat_type,
        }
    )
    assert "F_new_feature1" in beatinfo.features.keys()
    assert beatinfo.features["F_new_feature1"] == 890


def test_select_features(beatinfo):
    feats = ["feature1", "feature2"]
    beatinfo.select_features(feats)
    assert set(beatinfo.selected_features_names) == set(feats)
