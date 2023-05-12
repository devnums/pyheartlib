import pytest
import numpy
import pandas
from pyecg.data_arrhythmia import ArrhythmiaData, ECGSequence
from pyecg.io import load_data
import hashlib


@pytest.fixture
def arrhythmia_data():
    obj = ArrhythmiaData()
    return obj


@pytest.fixture
def record():
    r = {
        "signal": numpy.random.uniform(-1, 1, 230),
        "r_locations": [],
        "r_labels": [],
        "rhythms": ["(N", "(AFIB", "(T"],
        "rhythms_locations": [10, 100, 200],
    }
    return r


def test_full_annotate(arrhythmia_data, record):
    signal, full_ann = arrhythmia_data.full_annotate(record)
    assert len(signal) == len(full_ann)
    assert full_ann[0] == "unlab"
    assert full_ann[9] == "unlab"
    assert full_ann[10] == "(N"
    assert full_ann[99] == "(N"
    assert full_ann[100] == "(AFIB"
    assert full_ann[199] == "(AFIB"
    assert full_ann[200] == "(T"
    assert full_ann[-1] == "(T"


from dummy import DummyData

test_data_dir = "./tests/dummy_data"
dmm = DummyData(save_dir=test_data_dir)
dmm.save(record_name="dummy101")
dmm.save(record_name="dummy102")


@pytest.fixture
def arrhythmiadata():
    obj = ArrhythmiaData(base_path=test_data_dir, remove_bl=False, lowpass=False)
    return obj


def test_save_samples(arrhythmiadata):
    ann, sam = arrhythmiadata.save_samples(
        rec_list=["dummy101", "dummy102"],
        file_name="tmp.arr",
        win_size=400,
        stride=200,
    )
    annotated_records, samples_info = load_data(test_data_dir + "/tmp.arr")
    assert annotated_records[0]["rhythms"] == ann[0]["rhythms"]
    assert samples_info[0][2] == sam[0][2]


@pytest.fixture
def seq_generator1():
    obj = ArrhythmiaData(base_path=test_data_dir, remove_bl=False, lowpass=False)
    ann, sam = obj.save_samples(
        rec_list=["dummy101", "dummy102"],
        file_name="tmp.arr",
        win_size=400,
        stride=200,
    )
    obj = ECGSequence(
        ann,
        sam,
        class_labels=None,
        batch_size=23,
        raw=True,
        interval=75,
        shuffle=False,
        rri_length=7,
    )
    return obj


def test_getitem1(seq_generator1):
    batch = 0
    batch_label = seq_generator1.__getitem__(batch)[1]  # excerpt label
    batch_seq = seq_generator1.__getitem__(batch)[0][0]  # excerpt values
    batch_rri = seq_generator1.__getitem__(batch)[0][1]  # rr intervals
    batch_rri_feat = seq_generator1.__getitem__(batch)[0][2]  # calculated rri features
    assert batch_label.shape == (23,)
    assert batch_seq.shape == (23, 400)
    assert batch_label[0] == "(N"
    assert batch_label[2] == "(VT"
    assert batch_label[3] == "(VT"

    assert list(batch_rri[22]) == pytest.approx(
        [0.333, 0.166, 0.388, 0, 0, 0, 0], rel=0.01
    )
    assert list(batch_rri_feat[22]) == pytest.approx(
        [296, 94.4, 333, 222, 0.319, 194, 196, 0.663, 33.3], rel=0.01
    )


@pytest.fixture
def seq_generator2():
    obj = ArrhythmiaData(base_path=test_data_dir, remove_bl=False, lowpass=False)
    ann, sam = obj.save_samples(
        rec_list=["dummy101", "dummy102"],
        file_name="tmp.arr",
        win_size=400,
        stride=200,
    )
    obj = ECGSequence(
        ann,
        sam,
        class_labels=["(N", "(VT"],
        batch_size=23,
        raw=False,
        interval=75,
        shuffle=False,
        rri_length=7,
    )
    return obj


def test_getitem2(seq_generator2):
    batch = 0
    batch_label = seq_generator2.__getitem__(batch)[1]  # excerpt label
    batch_seq = seq_generator2.__getitem__(batch)[0][0]  # excerpt values
    batch_rri = seq_generator2.__getitem__(batch)[0][1]  # rr intervals
    batch_rri_feat = seq_generator2.__getitem__(batch)[0][2]  # calculated rri features
    assert batch_label.shape == (23,)
    assert batch_seq.shape == (23, 5, 14)
    assert batch_label[0] == 0
    assert batch_label[2] == 1
    assert batch_label[3] == 1

    assert list(batch_rri[22]) == pytest.approx(
        [0.333, 0.166, 0.388, 0, 0, 0, 0], rel=0.01
    )
    assert list(batch_rri_feat[22]) == pytest.approx(
        [296, 94.4, 333, 222, 0.319, 194, 196, 0.663, 33.3], rel=0.01
    )
