import pytest
import numpy
import pandas
from pyecg.data_beat import BeatData
from pyecg.beat_info import BeatInfo
from dummy import DummyData

test_data_dir = "./tests/dummy_data"
dmm = DummyData(save_dir=test_data_dir)
dmm.save(record_name="dummy101")
dmm.save(record_name="dummy102")


@pytest.fixture
def beatdata():
    obj = BeatData(
        base_path=test_data_dir,
        win=[300, 200],
        num_pre_rr=2,
        num_post_rr=1,
    )
    return obj


def test_make_frags(beatdata):
    rec_dict = beatdata.get_ecg_record(record_id="dummy101")
    signal, r_locations, r_labels, _, _ = rec_dict.values()
    signal_frags, beat_types, r_locs, s_idxs = beatdata.make_frags(
        signal, r_locations, r_labels
    )
    assert isinstance(signal_frags, numpy.ndarray)
    assert isinstance(beat_types, list)
    assert isinstance(r_locs, list)
    assert isinstance(s_idxs, list)
    assert len(signal_frags) > 0
    assert len(beat_types) == len(signal_frags)
    assert len(r_locs) == len(signal_frags)
    assert len(s_idxs) == len(signal_frags)
    assert signal_frags.shape[1] == sum(beatdata.win)
    assert len(r_locs[0]) == beatdata.num_pre_rr + beatdata.num_post_rr + 1
    assert beatdata.beat_loc == beatdata.num_pre_rr


def test_make_dataset(beatdata):
    data = beatdata.make_dataset(records=["dummy101", "dummy102"])
    assert isinstance(data, dict)
    expected_dict = {
        "waveforms": numpy.ndarray,
        "beat_feats": pandas.DataFrame,
        "labels": numpy.ndarray,
    }
    assert len(data.keys()) == len(expected_dict.keys())
    assert set(data.keys()) == set(expected_dict.keys())
    for key in expected_dict.keys():
        assert isinstance(data[key], expected_dict[key])
        assert len(data[key]) > 0
    assert len(data["waveforms"]) == len(data["labels"])
    assert data["waveforms"].shape[1] == sum(beatdata.win)


def test_beat_info_feat(beatdata):
    rec_dict = beatdata.get_ecg_record(record_id="dummy101")
    signal, r_locations, r_labels, _, _ = rec_dict.values()
    signal_frags, beat_types, r_locs, s_idxs = beatdata.make_frags(
        signal, r_locations, r_labels
    )
    rec_ids_list = [100] * len(s_idxs)

    beatinfo = BeatInfo(beat_loc=beatdata.beat_loc)
    beat_feats, labels = beatdata.beat_info_feat(
        {
            "waveforms": signal_frags.tolist(),
            "rpeak_locs": r_locs,
            "rec_ids": rec_ids_list,
            "start_idxs": s_idxs,
            "labels": beat_types,
        },
        beatinfo,
    )
    assert isinstance(beat_feats, list)
    assert isinstance(beat_feats[0], dict)
    assert isinstance(labels, list)
    assert len(labels) > 0


def test_report_stats(beatdata):
    train_labels = numpy.array(["N", "N", "V", "N", "A"])
    test_labels = numpy.array(["N", "V", "A", "V", "A", "V"])
    yds_list = [train_labels, test_labels]
    res = beatdata.report_stats(yds_list)

    assert isinstance(res, list)
    assert isinstance(res[0], dict)
    assert res[0]["N"] == 3
    assert res[1]["A"] == 2


def test_report_stats_table(beatdata):
    train_labels = numpy.array(["N", "N", "V", "N", "A"])
    test_labels = numpy.array(["N", "V", "A", "V", "A", "V"])
    yds_list = [train_labels, test_labels]
    res = beatdata.report_stats_table(yds_list, ["Train", "Test"])

    assert isinstance(res, pandas.DataFrame)
    assert res["N"]["Train"] == 3
    assert res["A"]["Test"] == 2
