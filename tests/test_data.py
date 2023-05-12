import pytest
import numpy
from pyecg.data_beat import BeatData
from dummy import DummyData


test_data_dir = "./tests/dummy_data"
dmm = DummyData(save_dir=test_data_dir)
dmm.save(record_name="dummy101")


beatdata = BeatData(
    base_path=test_data_dir,
    win=[300, 200],
    num_pre_rr=2,
    num_post_rr=5,
    remove_bl=True,
    lowpass=True
)


def test_get_ecg_record():
    # there must be a directory named data in the project dir.
    data = beatdata.get_ecg_record(record_id="dummy101")
    assert isinstance(data, dict)
    expected_dict = {
        "signal": numpy.ndarray,
        "r_locations": list,
        "r_labels": list,
        "rhythms": list,
        "rhythms_locations": list,
    }
    assert len(data.keys()) == len(expected_dict.keys())
    assert set(data.keys()) == set(expected_dict.keys())
    for key in expected_dict.keys():
        assert isinstance(data[key], expected_dict[key])
        assert len(data[key]) > 0
    assert len(data["r_locations"]) == len(data["r_labels"])
