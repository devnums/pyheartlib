import os
import numpy
from pyecg.io import get_data
from dummy import DummyData

test_data_dir = "./tests/dummy_data"
record_name = "dummy101"
dmm = DummyData(save_dir=test_data_dir)
dmm.save(record_name="dummy101")


def test_get_data():
    # there must be a directory named data in the project dir.
    record_path = os.path.join(os.getcwd(), test_data_dir, record_name)
    data = get_data(record_path)
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
