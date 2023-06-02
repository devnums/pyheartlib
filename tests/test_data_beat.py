import pytest
import numpy
import pandas
import os
from pyheartlib.data_beat import BeatData
from pyheartlib.beat_info import BeatInfo
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
    beatinfo = BeatInfo()
    ds = beatdata.make_dataset(["dummy101", "dummy102"], beatinfo)
    assert isinstance(ds, dict)
    expected_dict = {
        "waveforms": numpy.ndarray,
        "beat_feats": pandas.DataFrame,
        "labels": numpy.ndarray,
    }
    assert len(ds.keys()) == len(expected_dict.keys())
    assert set(ds.keys()) == set(expected_dict.keys())
    for key in expected_dict.keys():
        assert isinstance(ds[key], expected_dict[key])
        assert len(ds[key]) > 0
    assert len(ds["waveforms"]) == len(ds["labels"])
    assert ds["waveforms"].shape[1] == sum(beatdata.win)


def test_beat_info_feat(beatdata):
    rec_dict = beatdata.get_ecg_record(record_id="dummy101")
    signal, r_locations, r_labels, _, _ = rec_dict.values()
    signal_frags, beat_types, r_locs, s_idxs = beatdata.make_frags(
        signal, r_locations, r_labels
    )
    rec_ids_list = [100] * len(s_idxs)

    beatinfo = BeatInfo()
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


def test_save_dataset_inter(beatdata):
    file_name = "inter.beat"
    tmpfile = os.path.join(test_data_dir, file_name)
    if os.path.exists(tmpfile):
        os.remove(tmpfile)
    beatinfo = BeatInfo()
    beatdata.save_dataset_inter(["dummy101", "dummy102"], beatinfo, file_name)
    assert os.path.exists(tmpfile)


def test_save_dataset_intra(beatdata):
    file_train = "intra_train.beat"
    tmpfiletrain = os.path.join(test_data_dir, file_train)
    if os.path.exists(tmpfiletrain):
        os.remove(tmpfiletrain)
    file_test = "intra_test.beat"
    tmpfiletest = os.path.join(test_data_dir, file_test)
    if os.path.exists(tmpfiletest):
        os.remove(tmpfiletest)
    beatinfo = BeatInfo()
    beatdata.save_dataset_intra(["dummy101", "dummy102"], beatinfo)
    assert os.path.exists(tmpfiletrain)
    assert os.path.exists(tmpfiletest)


def test_save_dataset_single(beatdata):
    file_train = "dummy101_train.beat"
    tmpfiletrain = os.path.join(test_data_dir, file_train)
    if os.path.exists(tmpfiletrain):
        os.remove(tmpfiletrain)
    file_test = "dummy101_test.beat"
    tmpfiletest = os.path.join(test_data_dir, file_test)
    if os.path.exists(tmpfiletest):
        os.remove(tmpfiletest)
    beatinfo = BeatInfo()
    beatdata.save_dataset_single("dummy101", beatinfo)
    assert os.path.exists(tmpfiletrain)
    assert os.path.exists(tmpfiletest)


def test_load_data(beatdata):
    file_name = "load.beat"
    beatinfo = BeatInfo()
    beatdata.save_dataset_inter(["dummy101"], beatinfo, file_name)
    ds = beatdata.load_data(file_name)
    assert isinstance(ds, dict)
    assert len(ds) == 3


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


def test_per_record_stats(beatdata):
    df = beatdata.per_record_stats(rec_ids_list=["dummy101", "dummy101"])
    assert df["N"][0] == 15
    assert df["V"][0] == 1
    assert df["A"][0] == 1


def test_slice_data(beatdata):
    beatinfo = BeatInfo()
    ds = beatdata.make_dataset(["dummy101", "dummy102"], beatinfo)
    lbs = ["N", "V"]
    res = beatdata.slice_data(ds, lbs)
    assert ds.keys() == res.keys()
    for key in ds.keys():
        assert isinstance(ds[key], type(res[key]))
    res_ls, res_cnt = numpy.unique(res["labels"], return_counts=True)
    ds_ls, ds_cnt = numpy.unique(ds["labels"], return_counts=True)
    res_dict = dict(zip(list(res_ls), list(res_cnt)))
    ds_dict = dict(zip(list(ds_ls), list(ds_cnt)))
    assert list(res_ls) == lbs
    for lb in lbs:
        assert res_dict[lb] == ds_dict[lb]


def test_search_label(beatdata):
    beatinfo = BeatInfo()
    ds = beatdata.make_dataset(["dummy101"], beatinfo)
    indexes = beatdata.search_label(ds, "A")
    assert indexes == [2]
    indexes = beatdata.search_label(ds, "N")
    assert indexes == [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def test_clean_inf_nan(beatdata):
    beatinfo = BeatInfo()
    ds = beatdata.make_dataset(["dummy101"], beatinfo)
    ds["beat_feats"].iloc[2, 0] = numpy.inf
    ds["beat_feats"].iloc[3, 1] = numpy.nan
    clean_ds = beatdata.clean_inf_nan(ds)
    assert ds.keys() == clean_ds.keys()
    assert clean_ds["labels"].shape == tuple(numpy.subtract(ds["labels"].shape, (2,)))
    assert clean_ds["waveforms"].shape == tuple(
        numpy.subtract(ds["waveforms"].shape, (2, 0))
    )
    assert clean_ds["beat_feats"].shape == tuple(
        numpy.subtract(ds["beat_feats"].shape, (2, 0))
    )


def test_clean_IQR(beatdata):
    beatinfo = BeatInfo()
    ds = beatdata.make_dataset(["dummy101"], beatinfo)
    clean_ds = beatdata.clean_IQR(ds)
    assert ds.keys() == clean_ds.keys()
    assert clean_ds["waveforms"].shape[0] == clean_ds["labels"].shape[0]
    assert clean_ds["beat_feats"].shape[0] == clean_ds["labels"].shape[0]


def test_clean_IQR_class(beatdata):
    beatinfo = BeatInfo()
    ds = beatdata.make_dataset(["dummy101"], beatinfo)
    clean_ds = beatdata.clean_IQR_class(ds)
    assert ds.keys() == clean_ds.keys()
    assert clean_ds["waveforms"].shape[0] == clean_ds["labels"].shape[0]
    assert clean_ds["beat_feats"].shape[0] == clean_ds["labels"].shape[0]


def test_append_ds(beatdata):
    beatinfo = BeatInfo()
    ds = beatdata.make_dataset(["dummy101"], beatinfo)
    res = beatdata.append_ds(ds, ds)
    assert ds.keys() == res.keys()
    assert res["labels"].shape == tuple(numpy.multiply(ds["labels"].shape, (2,)))
    assert res["waveforms"].shape == tuple(
        numpy.multiply(ds["waveforms"].shape, (2, 1))
    )
    assert res["beat_feats"].shape == tuple(
        numpy.multiply(ds["beat_feats"].shape, (2, 1))
    )
    wshape = ds["waveforms"].shape
    assert res["waveforms"][wshape[0], 3] == ds["waveforms"][0, 3]
    fshape = ds["beat_feats"].shape
    assert res["waveforms"][fshape[0] + 2, 5] == ds["waveforms"][2, 5]
