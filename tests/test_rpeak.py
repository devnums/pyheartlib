import pytest
import numpy
import pandas
from pyecg.data_rpeak import RpeakData, ECGSequence
from pyecg.io import load_data
import hashlib


@pytest.fixture
def rpeak_data():
    obj = RpeakData()
    return obj


@pytest.fixture
def record():
    r = {
        "signal": numpy.random.uniform(-1, 1, 4030),
        "r_locations": [
            200,
            410,
            650,
            830,
            1020,
            1240,
            1650,
            1840,
            2015,
            2310,
            2620,
            2910,
            3200,
        ],
        "r_labels": ["N", "V", "A", "A", "N", "N", "V", "A", "A", "N", "N", "V", "A"],
        "rhythms": ["(N", "(AFIB", "(T"],
        "rhythms_locations": [10, 800, 1500],
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
    assert (set(full_ann)) == (set(rlabels))


from dummy import DummyData

test_data_dir = "./tests/dummy_data"
dmm = DummyData(save_dir=test_data_dir)
dmm.save(record_name="dummy101")
dmm.save(record_name="dummy102")


@pytest.fixture
def rpeakdata():
    obj = RpeakData(data_path=test_data_dir, remove_bl=False, lowpass=False)
    return obj


def test_save_samples(rpeakdata):
    ann, sam = rpeakdata.save_samples(
        rec_list=["dummy101", "dummy102"],
        file_name=test_data_dir + "/tmp.rpeak",
        win_size=400,
        stride=200,
    )
    annotated_records, samples_info = load_data(test_data_dir + "/tmp.rpeak")
    assert annotated_records[0]["r_labels"] == ann[0]["r_labels"]
    assert samples_info[0][2] == sam[0][2]


@pytest.fixture
def seq_generator1():
    obj = RpeakData(data_path=test_data_dir, remove_bl=False, lowpass=False)
    ann, sam = obj.save_samples(
        rec_list=["dummy101", "dummy102"],
        file_name=test_data_dir + "/tmp.rpeak",
        win_size=400,
        stride=200,
    )
    obj = ECGSequence(
        ann, sam, binary=False, batch_size=7, raw=True, interval=75, shuffle=False
    )
    return obj


def test_getitem1(seq_generator1):
    batch = 0
    seq = seq_generator1.__getitem__(batch)[0]  # excerpt values
    label = seq_generator1.__getitem__(batch)[1]  # excerpt label
    assert seq.shape == (7, 400)
    assert label.shape == (7, 6)
    assert int(label[0][0]) == 0
    assert label[0][1] == "N"
    assert int(label[0][2]) == 0
    assert int(label[0][3]) == 0
    assert int(label[0][4]) == 0
    assert int(label[0][5]) == 0
    assert int(label[1][0]) == 0
    assert label[1][3] == "N"
    assert int(label[1][5]) == 0
    assert label[4][0] == "V"
    assert label[6][4] == "A"


@pytest.fixture
def seq_generator2():
    obj = RpeakData(data_path=test_data_dir, remove_bl=False, lowpass=False)
    ann, sam = obj.save_samples(
        rec_list=["dummy101", "dummy102"],
        file_name=test_data_dir + "/tmp.rpeak",
        win_size=400,
        stride=200,
    )
    obj = ECGSequence(
        ann, sam, binary=True, batch_size=7, raw=True, interval=75, shuffle=False
    )
    return obj


def test_getitem2(seq_generator2):
    batch = 0
    seq = seq_generator2.__getitem__(batch)[0]  # excerpt values
    label = seq_generator2.__getitem__(batch)[1]  # excerpt label
    assert seq.shape == (7, 400)
    assert label.shape == (7, 6)
    assert label[0][0] == 0
    assert label[0][1] == 1
    assert label[0][2] == 0
    assert label[0][3] == 0
    assert label[0][4] == 0
    assert label[0][5] == 0
    assert label[1][0] == 0
    assert label[1][3] == 1
    assert label[1][5] == 0
    assert label[4][0] == 1
    assert label[6][4] == 1


@pytest.fixture
def seq_generator3():
    obj = RpeakData(data_path=test_data_dir, remove_bl=False, lowpass=False)
    ann, sam = obj.save_samples(
        rec_list=["dummy101", "dummy102"],
        file_name=test_data_dir + "/tmp.rpeak",
        win_size=400,
        stride=200,
    )
    obj = ECGSequence(
        ann,
        sam,
        class_labels=[0, "N", "V", "A"],
        batch_size=7,
        raw=True,
        interval=75,
        shuffle=False,
    )
    return obj


def test_getitem3(seq_generator3):
    batch = 0
    seq = seq_generator3.__getitem__(batch)[0]  # excerpt values
    label = seq_generator3.__getitem__(batch)[1]  # excerpt label
    assert seq.shape == (7, 400)
    assert label.shape == (7, 6)
    assert label[0][0] == 0
    assert label[0][1] == 1
    assert label[0][2] == 0
    assert label[0][3] == 0
    assert label[0][4] == 0
    assert label[0][5] == 0
    assert label[1][0] == 0
    assert label[1][3] == 1
    assert label[1][5] == 0
    assert label[4][0] == 2
    assert label[6][4] == 3
