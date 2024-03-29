#############################################################################
# Copyright (c) 2023 Pyheartlib team. - All Rights Reserved                 #
# Project repo: https://github.com/devnums/pyheartlib                       #
# Contact: devnums.code@gmail.com                                           #
#                                                                           #
# This file is part of the Pyheartlib project.                              #
# To see the complete LICENSE file visit:                                   #
# https://github.com/devnums/pyheartlib/blob/main/LICENSE                   #
#############################################################################


import os

import numpy
import yaml
from dummy import DummyData

from pyheartlib.io import get_data

test_data_dir = "./tests/dummy_data"
record_name = "dummy101"
dmm = DummyData(save_dir=test_data_dir)
dmm.save(record_name="dummy101")

conf_path = os.path.join(os.getcwd(), "tests/dummy_data/config.yaml")
with open(conf_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def test_get_data():
    # there must be a directory named data in the project dir.
    record_path = os.path.join(os.getcwd(), test_data_dir, record_name)
    data = get_data(record_path, config)
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
