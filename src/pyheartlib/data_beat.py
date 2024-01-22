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

import numpy as np
import pandas as pd
from tqdm import tqdm

from pyheartlib.data import Data
from pyheartlib.io import load_data, save_data


class BeatData(Data):
    """
    Processes the provided ECG records and creates a dataset containing
    waveforms, features, and annotations.

    Parameters
    ----------
    base_path : str, optional
        Path of the main directory for storing the original and
        processed data, by default None
    win : list, optional
        [Onset, Offset] of signal excerpts around the R-peaks,
        by default [60, 120]
    num_pre_rr : int, optional
        Number of preceding R-peak locations to be included for each beat,
        by default 10
    num_post_rr : int, optional
        Number of subsequent R-peak locations to be included for each beat,
        by default 10
    remove_bl : bool, optional
        If True, the baseline wander is removed from the original signals
        prior to extracting excerpts, by default False
    lowpass : bool, optional
        Whether or not to apply low-pass filter to the original signals,
        by default False
    cutoff : int, optional
        Parameter of the low pass-filter, by default 45
    order : int, optional
        Parameter of the low pass-filter, by default 15
    progress_bar : bool, optional
        Whether to display a progress bar, by default True
    processors : list, optional
        Ordered list of functions' names for preprocessing the raw signals.
        Each function takes a one-dimensional NumPy array as its input and
        returns an array of the same length.

    Examples
    --------
    >>> beatdata = BeatData(base_path="./data", win=[200, 200],
    >>>                     remove_bl=False, lowpass=False,
    >>>                     progress_bar=True)
    >>> # create a BeatInfo object
    >>> beatinfo = BeatInfo()
    >>> # save the dataset file
    >>> beatdata.save_dataset_inter(DS1[17:18], beatinfo, file="train.beat")
    >>> # load the dataset from file
    >>> train_ds = beatdata.load_data(file_name="train.beat")
    File loaded from: ./data/train.beat
    -Shape of "waveforms" is (2985, 400). Number of samples is 2985.
    -Shape of "beat_feats" is (2985, 27). Number of samples is 2985.
    -Shape of "labels" is (2985,). Number of samples is 2985.
                N  L  R  j  e  V  E    A  S  a  J  F  f  /  Q
    train.beat  2601  0  0  0  0  1  0  383  0  0  0  0  0  0  0
    """

    def __init__(
        self,
        base_path=None,
        win=[60, 120],
        num_pre_rr=10,
        num_post_rr=10,
        remove_bl=False,
        lowpass=False,
        cutoff=45,
        order=15,
        progress_bar=True,
        **kwargs,
    ):
        super().__init__(
            base_path,
            remove_bl,
            lowpass,
            cutoff,
            order,
            **kwargs,
        )
        self.syms = self.config["BEAT_TYPES"]
        self.win = win
        self.num_pre_rr = num_pre_rr
        self.num_post_rr = num_post_rr
        self.beat_loc = self.num_pre_rr
        self.progress_bar = not progress_bar

    def make_frags(
        self,
        signal,
        r_locations=None,
        r_label=None,
    ):
        """Fragments one signal into beats and returns the signal excerpts
        and corresponding labels.

        Parameters
        ----------
        signal : list
            A list containing signal values.
        r_locations : list
            A list containing rpeak locations on the signal.
        r_label : list
            A list containing the rpeak(beat) labels.

        Returns
        -------
        Tuple
            signal_frags : numpy.ndarray
                A 2D array containing extracted beat excerpts.
            beat_types : list
                Contains the corresponding labels of each beat excerpt.
            r_locs : list
                A list containing lists of previous, itself, and future rpeak
                locations for each beat. Can be used for HRV calculations.
            s_idxs : list
                Contains the starting point of each extracted beat excerpt
                on the original signal. This is computed by subtracting the
                window onset from the rpeak location.
        """
        if not self.num_pre_rr < (len(r_locations) - self.num_post_rr):
            raise ValueError(
                "Wrong num_pre_rr and num_post_rr in \
                respect to the number of peaks!"
            )

        win = self.win
        frags = []
        beat_types = []
        r_locs = []
        s_idxs = []  # start indexes
        for i in range(self.num_pre_rr, (len(r_locations) - self.num_post_rr)):
            start_idx = r_locations[i] - win[0]
            end_idx = r_locations[i] + win[1]
            seg = signal[start_idx:end_idx]
            if len(seg) == win[0] + win[1]:
                frags.append(seg)
                beat_types.append(r_label[i])
                r_locs.append(
                    list(
                        r_locations[
                            i - self.num_pre_rr : i + self.num_post_rr + 1
                        ]
                    )
                )
                s_idxs.append(start_idx)
        signal_frags = np.array(frags)
        return signal_frags, beat_types, r_locs, s_idxs

    def make_dataset(self, records, beatinfo_obj=None):
        """Creates a dataset from the provided records.

        Parameters
        ----------
        records : list
            A list containing records ids.
        beatinfo_obj : instance of BeatInfo.

        Returns
        -------
        dict
            Dictionary with keys:

            'waveforms' : numpy.ndarray
                2D array of beat waveforms.
            'beat_feats' : pd.DataFrame
                DataFrame of beats' features.
            'labels' : numpy.ndarray
                1D array of beats' labels.
        """
        xds = []
        yds = []
        rds = []
        start_idxs = []
        rec_ids_list = []  # debug
        for i in tqdm(range(len(records)), disable=self.progress_bar):
            rec_dict = self.get_ecg_record(record_id=records[i])
            signal, r_locations, r_labels, _, _ = rec_dict.values()
            signal_frags, beat_types, r_locs, s_idxs = self.make_frags(
                signal, r_locations, r_labels
            )
            start_idxs.extend(s_idxs)
            # adding record id for each excerpt
            rec_ids_list.extend([records[i]] * len(start_idxs))
            xds = xds + signal_frags.tolist()
            yds = yds + beat_types
            rds = rds + r_locs
            i += 1
        xds = np.array(xds)
        if beatinfo_obj is not None:
            beat_feats, labels = self.beat_info_feat(
                {
                    "waveforms": xds,
                    "rpeak_locs": rds,
                    "rec_ids": rec_ids_list,
                    "start_idxs": start_idxs,
                    "labels": yds,
                },
                beatinfo_obj,
            )
            # beat_feats = [
            #   [i for i in beat_feat.values()] for beat_feat in beat_feats]
        else:
            beat_feats = [{"empty": ""}]
            labels = yds

        return {
            "waveforms": np.array(xds),
            "beat_feats": pd.DataFrame(beat_feats),
            "labels": np.array(labels),
        }

    def beat_info_feat(self, data, beatinfo_obj):
        """Provides the computed features for all the beats.

        Parameters
        ----------
        data : dict

            Dictionary with keys:

            'waveform' : numpy.ndarray
                Array of waveforms (#waveforms, len_waveforms, #channels).
            'rpeak_locs' : list
                List of rpeak locations.
            'rec_ids' : list
                List of record ids.
            'start_idxs' : list
                List of start_idxs of waveforms on the raw signal.
            'labels' : list
                List of beat labels.
        beatinfo_obj : Instance of BeatInfo

        Returns
        -------
        Tuple
            features : list
                Contains feature dictionaries for all the beats.
            labels : list
                Contains corresponding beat labels.
        """
        features = []
        labels = []
        beatinfo_obj.fs = self.sampling_rate
        beatinfo_obj.beat_loc = self.beat_loc
        # beats loop
        for i in tqdm(
            range(len(data["rpeak_locs"])), disable=self.progress_bar
        ):
            beatinfo_obj(
                {
                    "waveform": data["waveforms"][i],
                    "rpeak_locs": data["rpeak_locs"][i],
                    "rec_id": data["rec_ids"][i],
                    "start_idx": data["start_idxs"][i],
                    "label": data["labels"][i],
                }
            )
            # check if the beat correctly annotated
            if (beatinfo_obj.bwaveform) is None:
                continue
            # feature dict of each beat
            beat_features_dict = beatinfo_obj.features
            # append for each beat
            features.append(beat_features_dict)
            # only append labels for beats that are correct
            labels.append(beatinfo_obj.label)
            # import time
            # time.sleep(0.3)
        return features, labels

    def save_dataset_inter(self, records, beatinfo_obj, file=None):
        """Creates a dataset from the given record IDs.

        Parameters
        ----------
        records : list
            List of records IDs.
        beatinfo_obj : Instance of BeatInfo.
        file : str
            Name of the file that will be saved.
        """

        if file is None:
            raise ValueError("Save file path is not provided!")
        ds = self.make_dataset(records, beatinfo_obj)
        save_data(ds, file_path=os.path.join(self.base_path, file))

    def save_dataset_intra(
        self,
        records,
        beatinfo_obj,
        split_ratio=0.3,
        file_prefix="intra",
    ):
        """Makes the dataset in intra-patient way.

        Parameters
        ----------
        records : list, optional
            List of records IDs.
        beatinfo_obj : Instance of BeatInfo.
        split_ratio : float, optional
            Ratio of test set, by default 0.3
        file_prefix : str, optional
            Prefix for the file names to be saved, by default 'intra'
        """
        xdata_train = []
        fdata_train = pd.DataFrame()
        ydata_train = []
        xdata_test = []
        fdata_test = pd.DataFrame()
        ydata_test = []
        for record in records:
            ds = self.make_dataset([record], beatinfo_obj)

            # slice the dataset based on the split_ratio
            xarr = ds["waveforms"]
            farr = ds["beat_feats"]
            yarr = ds["labels"]

            split_idx = int(split_ratio * xarr.shape[0])
            xarr_train = xarr[:split_idx]
            xarr_test = xarr[split_idx:]
            farr_train = farr[:split_idx]
            farr_test = farr[split_idx:]
            yarr_train = yarr[:split_idx]
            yarr_test = yarr[split_idx:]

            xdata_train = xdata_train + xarr_train.tolist()
            fdata_train = pd.concat([fdata_train, farr_train])
            ydata_train = ydata_train + yarr_train.tolist()
            xdata_test = xdata_test + xarr_test.tolist()
            fdata_test = pd.concat([fdata_test, farr_test])
            ydata_test = ydata_test + yarr_test.tolist()

        xdata_train = np.array(xdata_train)
        ydata_train = np.array(ydata_train)
        xdata_test = np.array(xdata_test)
        ydata_test = np.array(ydata_test)

        ds_train = {
            "waveforms": xdata_train,
            "beat_feats": fdata_train,
            "labels": ydata_train,
        }
        ds_test = {
            "waveforms": xdata_test,
            "beat_feats": fdata_test,
            "labels": ydata_test,
        }

        file_train = file_prefix + "_train" + ".beat"
        save_data(ds_train, file_path=os.path.join(self.base_path, file_train))

        file_test = file_prefix + "_test" + ".beat"
        save_data(ds_test, file_path=os.path.join(self.base_path, file_test))

    def save_dataset_single(
        self, record, beatinfo_obj, split_ratio=0.3, file=None
    ):
        """Saves the signal fragments and their labels into a file for
        a single record.

        Parameters
        ----------
        record : str
            Record id.
        beatinfo_obj : instance of BeatInfo
        split_ratio : float, optional
            Ratio of test set, by default 0.3
        file : str, optional
            Name of the file to be saved, by default None
        """
        ds = self.make_dataset([record], beatinfo_obj)

        # slice the dataset based on the split_ratio
        xarr = ds["waveforms"]
        farr = ds["beat_feats"]
        yarr = ds["labels"]
        split_idx = int(split_ratio * xarr.shape[0])
        xarr_train = xarr[:split_idx]
        xarr_test = xarr[split_idx:]
        farr_train = farr[:split_idx]
        farr_test = farr[split_idx:]
        yarr_train = yarr[:split_idx]
        yarr_test = yarr[split_idx:]

        ds_train = {
            "waveforms": xarr_train,
            "beat_feats": farr_train,
            "labels": yarr_train,
        }
        ds_test = {
            "waveforms": xarr_test,
            "beat_feats": farr_test,
            "labels": yarr_test,
        }
        if file is None:
            file_train = str(record) + "_train" + ".beat"
            file_test = str(record) + "_test" + ".beat"
        else:
            file_train = file + "_train" + ".beat"
            file_test = file + "_test" + ".beat"
        save_data(ds_train, file_path=os.path.join(self.base_path, file_train))
        save_data(ds_test, file_path=os.path.join(self.base_path, file_test))

    def load_data(self, file_name):
        """Loads a file containing a dataframe.

        Parameters
        ----------
        file_name : str
            File name. The final final path is the join of
            base path and file name.

        Returns
        -------
        dict
            Dataset with keys: "waveforms", "beat_feats", and "labels".
        """

        if file_name is None:
            raise ValueError("Load file name is not provided!")
        ds = load_data(file_path=os.path.join(self.base_path, file_name))
        for k, v in ds.items():
            print(
                '-Shape of "{}" is {}. Number of samples is {}.'.format(
                    k, v.shape, v.shape[0]
                )
            )
        print(self.report_stats_table([ds["labels"]], [file_name]))
        print()
        return ds

    def report_stats(self, yds_list):
        """Counts the number of samples for each label type in the data.

        Parameters
        ----------
        yds_list : list
            List containing several label data sets. e.g train,val,test.

        Returns
        -------
        list
            A list of dictionaries. One dictionary per one data set.
            Keys are labels types(symbols) and values are the counts of
            each specific symbol.
        """
        res_list = []
        for yds in yds_list:
            res = {}
            for sym in self.syms:
                indexes = [i for i, item in enumerate(yds) if item == sym]
                res[sym] = len(indexes)
            res_list.append(res)
        return res_list

    def report_stats_table(self, yds_list, name_list=[]):
        """Returns the number of samples for each label type in the data.

        Parameters
        ----------
        yds_list : list
            List containing several label data. e.g train,val,test.
        name_list : list, optional
            A list of strings as the name of label data e.g. train,val,test,
            by default []

        Returns
        -------
        pandas.dataframe
            A dataframe containing symbols and their counts.
        """
        if len(name_list) == len(yds_list):
            indx = name_list
        else:
            indx = None
        res_list = self.report_stats(yds_list)
        df = pd.DataFrame(res_list, index=indx)
        return df

    def per_record_stats(self, rec_ids_list=None, cols=None):
        """Returns a dataframe containing the number of
        each type in each record.

        Parameters
        ----------
        rec_ids_list : list, optional
            List of record ids, by default None
        cols : list
            List of labels classes, by default None

        Returns
        -------
        pandas.dataframe
            Contains count of each label type.
        """

        if cols is None:
            cols = self.syms
        ld = []
        for rec_id in rec_ids_list:
            rec_dict = self.get_ecg_record(record_id=rec_id)
            res = np.unique(rec_dict["r_labels"], return_counts=True)
            ld.append(dict(zip(res[0], res[1])))

        df = pd.DataFrame(ld, index=rec_ids_list)
        df.fillna(0, inplace=True)
        df = df.astype("int32")
        df = df[list(set(cols) & set(df.columns))]
        return df

    def slice_data(self, ds, labels):
        """Returns the data according to the provided annotation list.

        Parameters
        ----------
        ds : dics
            Dataset with keys: "waveforms", "beat_feats", and "labels".
        labels : list
            List of labels to be kept in the output.

        Returns
        -------
        dict
            Dataset with keys: "waveforms", "beat_feats", and "labels".

        """

        sliced_y = ds["labels"]
        sliced_x = ds["waveforms"]
        sliced_f = ds["beat_feats"]
        indexes_keep = []
        for sym in labels:
            inds = [i for i, item in enumerate(sliced_y) if item == sym]
            indexes_keep.extend(inds)
        sliced_y = sliced_y[indexes_keep]
        sliced_x = sliced_x[indexes_keep]
        sliced_f = sliced_f.iloc[indexes_keep]
        return {
            "waveforms": sliced_x,
            "beat_feats": sliced_f,
            "labels": sliced_y,
        }

    def search_label(self, inp, sym="N"):
        """Searches the provided data and returns the
        indexes for a patricular label.

        Parameters
        ----------
        inp : dict or numpy.ndarray
            Input can be a dictionary having a 'labels' key, or
            a 1D numpy array containing labels.
        sym : str, optional
            The label to be searched for in the dataset, by default 'N'

        Returns
        -------
        list
            A list of indexes corresponding to the searched label.

        Raises
        ------
        TypeError
            Input data must be a dictionary or a numpy array.
        """

        if isinstance(inp, dict):
            yds = list(inp["labels"])
        elif isinstance(inp, np.ndarray):
            yds = inp
        else:
            raise TypeError("input must be a dictionary or a 1D numpy array!")
        indexes = [i for i, item in enumerate(yds) if item == sym]
        return indexes

    def clean_inf_nan(self, ds):
        """Cleans the dataset by removing samples (rows) with inf or nan in
        computed features.

        Parameters
        ----------
        ds : dict
            Dataset with keys: "waveforms", "beat_feats", and "labels".

        Returns
        -------
        dict
            Cleaned dataset with keys: "waveforms",
            "beat_feats", and "labels".

        """

        yds = ds["labels"]
        xds = ds["waveforms"]
        fds = ds["beat_feats"]
        # clean feature array
        indexes = []
        fds = fds.reset_index(drop=True)
        for indx, row in fds.iterrows():
            for label, content in row.items():
                if np.isinf(content).any() or np.isnan(content).any():
                    # fds[label][indx] = np.nan
                    indexes.append(indx)
        indexes = list(set(indexes))
        fds = fds.drop(indexes, axis=0, inplace=False)
        fds = fds.reset_index(drop=True)
        xds = np.delete(xds, indexes, axis=0)
        yds = np.delete(yds, indexes, axis=0)
        return {"waveforms": xds, "beat_feats": fds, "labels": yds}

    def clean_IQR(self, ds, factor=1.5, return_indexes=False):
        """Cleans the dataset by removing outliers using IQR method.

        Parameters
        ----------
        ds : dict
            Dataset with keys: "waveforms", "beat_feats", and "labels".
        factor : float, optional
            Parameter of IQR method, by default 1.5
        return_indexes : bool, optional
            If True returns indexes of outliers, otherwise returns
            cleaned dataset, by default False

        Returns
        -------
        dict or list
            Cleaned dataset with keys: "waveforms", "beat_feats",
            and "labels", or indexes of outliers.

        """

        yds = ds["labels"]
        xds = ds["waveforms"]
        fds = ds["beat_feats"]
        fds.reset_index(drop=True, inplace=True)
        # cleans a 2d array. Each column is a features, rows are samples.
        # Only fds.
        unmodified = []
        ind_outliers = []
        for i in range(fds.shape[1]):
            cont = fds.iloc[:, i]
            try:
                float(cont[0])
            except TypeError:
                unmodified.append(cont.name)
                continue
            Q1 = np.quantile(cont, 0.25, axis=0)
            Q3 = np.quantile(cont, 0.75, axis=0)
            IQR = Q3 - Q1
            inds = np.where(
                (cont > (Q3 + factor * IQR)) | (cont < (Q1 - factor * IQR))
            )[0]
            ind_outliers.extend(inds)
        ind_outliers = list(set(ind_outliers))
        fds = fds.drop(ind_outliers, axis=0, inplace=False)
        fds.reset_index(drop=True, inplace=True)
        xds = np.delete(xds, ind_outliers, axis=0)
        yds = np.delete(yds, ind_outliers, axis=0)
        print(f"{len(ind_outliers)} rows were removed!")
        print("The following columns were not considered: ")
        print(unmodified)
        if return_indexes is False:
            return {"waveforms": xds, "beat_feats": fds, "labels": yds}
        else:
            return ind_outliers

    def clean_IQR_class(self, ds, factor=1.5):
        """Cleans dataset by IQR method for every class separately.

        Parameters
        ----------
        ds : dict
            Dataset with keys: "waveforms", "beat_feats", and "labels".
        factor : float, optional
            Parameter of IQR method, by default 1.5

        Returns
        -------
        dict
            Cleaned dataset with keys: "waveforms",
            "beat_feats", and "labels".

        """

        for label in list(np.unique(ds["labels"])):
            sliced = self.slice_data(ds, [label])
            cleaned = self.clean_IQR(sliced, factor=factor)
            try:
                ds_all = self.append_ds(ds_all, cleaned)
            except NameError:
                ds_all = cleaned
        return ds_all

    def append_ds(self, ds1, ds2):
        """Appends two datasets.

        Parameters
        ----------
        ds1, ds2 : dict
            Datasets with keys: "waveforms", "beat_feats", and "labels".

        Returns
        -------
        dict
            Dataset with keys: "waveforms", "beat_feats", and "labels".

        """

        dss = dict()
        dss["waveforms"] = np.vstack((ds1["waveforms"], ds2["waveforms"]))
        dss["beat_feats"] = pd.concat([ds1["beat_feats"], ds2["beat_feats"]])
        dss["labels"] = np.vstack(
            (ds1["labels"].reshape(-1, 1), ds2["labels"].reshape(-1, 1))
        ).flatten()
        return dss
