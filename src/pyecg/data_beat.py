import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyecg.data import Data
from pyecg.dataset_config import DATA_DIR, DS1, MAP_AAMI, RECORDS
from pyecg.io import get_data, save_data, load_data


class BeatData(Data):
    """
    Processes the provided data and creates the dataset file containg waveforms,features,and labels.

    Examples
    --------
    Creating the object:
    >>> beatdata = BeatData(base_path='../data', win=[500,500],remove_bl=False,lowpass=False)
    Saving the dataset file:
    >>> beatdata.save_dataset(records=DS1[:18], save_file_name='train.beat')
    Loading the dataset file into a dictionary:
    >>> ds = beatdata.load_data(file_name='train.beat')
            file loaded: ../data/train.beat
            shape of "waveforms" is (38949, 1000)
            shape of "beat_feats" is (38949, 62)
            shape of "labels" is (38949,)
    >>> x_train, r_train, y_train = ds.values()
    """

    def __init__(
        self,
        base_path=os.getcwd(),  # TODO look or create data dir
        data_path=DATA_DIR,
        win=[60, 120],
        num_pre_rr=10,
        num_post_rr=10,
        remove_bl=False,
        lowpass=False,
        sampling_rate=360,
        cutoff=45,
        order=15,
    ):
        """
        Parameters
        ----------
        base_path : str, optional
            Path of main directory for loading and saving data, by default os.getcwd()
        win : list, optional
            [onset,offset] of signal excerpts around the rpeaks, by default [60, 120]
        num_pre_rr : int, optional
            Number of previous rpeak locations must be returned by the function, by default 10
        num_post_rr : int, optional
            Number of future rpeak locations must be returned by the function., by default 10
        remove_bl : bool, optional
            If True removes the baseline from the raw signals before extracting beat excerpts, by default False
        lowpass : bool, optional
            Whether to apply low pass filtering to the raw signals, by default False
        sampling_rate : int, optional
            Sampling rate of the signals, by default 360
        cutoff : int, optional
            Parameter of the low pass filter, by default 45
        order : int, optional
            Parameter of the low pass filter, by default 15
        """
        super().__init__(
            base_path,
            data_path,
            remove_bl,
            lowpass,
            sampling_rate,
            cutoff,
            order,
        )
        self.syms = [k for k, v in MAP_AAMI.items()]
        self.win = win
        self.num_pre_rr = num_pre_rr
        self.num_post_rr = num_post_rr
        self.beat_loc = self.num_pre_rr

    def make_frags(
        self,
        signal,
        r_locations=None,
        r_label=None,
    ):
        """Fragments one signal into beats and returns the signal excerpts and their labels.

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
        signal_frags : numpy.ndarray
            A 2D array containing extracted beat excerpts.
        beat_types : list
            Contains the corersponding labels of each beat excerpt.
        r_locs : list
            A list containing lists of previous, itself, and future rpeak locations
            for each beat. Can be used for HRV calculations.
        s_idxs : list
            Contains the starting point of each extracted beat excerpt on the origional signal.
            This is computed by subtracting the window onset from the rpeak location.
        """
        if not self.num_pre_rr < (len(r_locations) - self.num_post_rr):
            raise ValueError(
                "Wrong num_pre_rr and num_post_rr in respect to the number of peaks!"
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
                    list(r_locations[i - self.num_pre_rr : i + self.num_post_rr + 1])
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

        Returns
        -------
        dict

            Keys are:

                'waveforms' : numpy.ndarray
                    2D ndarray of beat waveforms.
                'beat_feats' : pd.DataFrame
                    DataFrame of beats' features.
                'labels' : numpy.ndarray
                    1D ndarray of beats' labels.
        """
        xds = []
        yds = []
        rds = []
        start_idxs = []
        rec_ids_list = []  # debug
        for i in tqdm(range(len(records))):
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
            # beat_feats = [[i for i in beat_feat.values()] for beat_feat in beat_feats]
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
            Keys:
                'waveform' : list of waveforms
                'rpeak_locs' : list of rpeak locations
                'rec_ids' : list of record ids
                'start_idxs' : list of start_idxs of waveforms on the raw signal
                'labels' : list of beat labels
        Returns
        -------
        features : list
            A list containing feature dictionaries for all beats.
        labels : list
            Contains corresponding beat labels.
        """
        features = []
        labels = []
        # beats loop
        for i in tqdm(range(len(data["rpeak_locs"]))):
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

    def clean_irrelevant_data(self, ds):
        """Removes data with irrelevant labels(symbols) which are not in the symbols list.

        Parameters
        ----------
        ds : dict
            Dataset as a dictionary containing waveforms,beat features, and labels.

        Returns
        -------
        dict
            Cleaned dataset.
        """

        yds = ds["labels"]
        xds = ds["waveforms"]
        rds = ds["beat_feats"]
        indexes_rm = [i for i, item in enumerate(yds) if item not in self.syms]
        # indexes_rm = np.where(np.invert(np.isin(yds, self.syms)))[0]
        xds = np.delete(xds, indexes_rm, axis=0)
        rds = np.delete(rds, indexes_rm, axis=0)
        yds = np.delete(yds, indexes_rm, axis=0)
        # ydsc = [it for ind,it in enumerate(yds) if ind not in indexes_rm]
        return {"waveforms": xds, "beat_feats": rds, "labels": yds}

    def search_label(self, inp, sym="N"):
        """Searches the provided data and returns the indexes for a patricular label.

        Parameters
        ----------
        inp : dict or numpy.ndarray
            Input can be a dictionary having a 'labels' key, or a 1D numpy array.
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
            raise TypeError("input must be a dictionary or a numpy array!")
        indexes = [i for i, item in enumerate(yds) if item == sym]
        return indexes

    def report_stats(self, yds_list):
        """Counts the number of samples for each label type in the data.

        Parameters
        ----------
        yds_list : list
            List containing several label data sets. e.g train,val,test.

        Returns
        -------
        list
            A list of dictionaries. One dictionary per one data set. keys are labels types(symbols) and
            values are the counts of each specific symbol.
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
        """Returns the number of samples for each label type in the data as a dataframe.

        Parameters
        ----------
        yds_list : list
            List containing several label data. e.g train,val,test.
        name_list : list, optional
            A list of strings as the name of label data e.g. train,val,test, by default []

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

    def save_dataset_inter(
        self, records, beatinfo_obj, save_file_name=None, clean=True
    ):
        """Makes dataset and saves it in a file.

        Parameters
        ----------
        save_file_name : str
            Name of the file.
        records : list
            A list containing records ids.
        clean : bool, optional
            If True removes irrelavant label types, by default True

        Raises
        ------
        ValueError
            File path is not provided.
        """

        if save_file_name is None:
            raise ValueError("Save file path is not provided!")
        ds = self.make_dataset(records, beatinfo_obj)
        if clean == True:
            ds = self.clean_irrelevant_data(ds)
        save_data(ds, file_path=os.path.join(self.base_path, save_file_name))

    def save_dataset_single(
        self, record, clean=True, split_ratio=0.3, save_file_name=None
    ):
        """Saves the signal fragments and their labels into a file for a single record.

        Parameters
        ----------
        record : str
            Record id.
        clean : bool, optional
            If True doesnt include irrelevant label classes, by default True
        split_ratio : float, optional
            Ratio of test set, by default 0.3
        save_file_name : str, optional
            File name, by default None
        """
        ds = self.make_dataset([record], beatinfo_obj)
        if clean == True:
            ds = self.clean_irrelevant_data(ds)

        # slice the dataset based on the split_ratio
        xarr = ds["waveforms"]
        farr = ds["beat_feats"]
        yarr = ds["labels"]
        print(xarr.shape[0])
        print(len(xarr))
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
        ds_test = {"waveforms": xarr_test, "beat_feats": farr_test, "labels": yarr_test}
        if save_file_name == None:
            file_train = str(record) + "_train" + ".beat"
            file_test = str(record) + "_test" + ".beat"
        else:
            file_train = save_file_name + "_train" + ".beat"
            file_test = save_file_name + "_test" + ".beat"
        save_data(ds_train, file_path=os.path.join(self.base_path, file_train))
        save_data(ds_test, file_path=os.path.join(self.base_path, file_test))

    def save_dataset_intra(
        self,
        records,
        beatinfo_obj,
        clean=True,
        split_ratio=0.3,
        save_file_prefix="intra",
    ):
        """Makes and saves the dataset in intra way.

        Parameters
        ----------
        records : list, optional
                A list of record ids.
        clean : bool, optional
                If True doesnt include irrelevant label classes, by default True
        split_ratio : float, optional
                Ratio of test set, by default 0.3
        save_file_prefix : str, optional
                File name prefix, by default 'intra'
        """
        xdata_train = []
        fdata_train = []
        ydata_train = []
        xdata_test = []
        fdata_test = []
        ydata_test = []
        for record in records:
            ds = self.make_dataset(records=[record])
            if clean is True:
                ds = self.clean_irrelevant_data(ds)

            # slice the dataset based on the split_ratio
            xarr = ds["waveforms"]
            farr = ds["beat_feats"]
            yarr = ds["labels"]
            print(xarr.shape[0])
            print(len(xarr))
            split_idx = int(split_ratio * xarr.shape[0])
            xarr_train = xarr[:split_idx]
            xarr_test = xarr[split_idx:]
            farr_train = farr[:split_idx]
            farr_test = farr[split_idx:]
            yarr_train = yarr[:split_idx]
            yarr_test = yarr[split_idx:]

            xdata_train = xdata_train + xarr_train.tolist()
            fdata_train = fdata_train + farr_train.tolist()
            ydata_train = ydata_train + yarr_train.tolist()
            xdata_test = xdata_test + xarr_test.tolist()
            fdata_test = fdata_test + farr_test.tolist()
            ydata_test = ydata_test + yarr_test.tolist()

        xdata_train = np.array(xdata_train)
        fdata_train = np.array(fdata_train)
        ydata_train = np.array(ydata_train)
        xdata_test = np.array(xdata_test)
        fdata_test = np.array(fdata_test)
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

        file_train = save_file_prefix + "_train" + ".beat"
        save_data(ds_train, file_path=os.path.join(self.base_path, file_train))

        file_test = save_file_prefix + "_test" + ".beat"
        save_data(ds_test, file_path=os.path.join(self.base_path, file_test))

    def load_data(self, file_name):
        """Loads a file containing a dataframe.

        Parameters
        ----------
        file_name : str
            File name. The final final path is the join of base path and file name.

        Returns
        -------
        pandas.dataframe
            Dataset as a dataframe. Keys are waveforms,beat_feats,and labels. Values are numpy.ndarray.

        Raises
        ------
        ValueError
            File name is not provided.
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
        return ds

    def per_record_stats(self, rec_ids_list=DS1, cols=None):
        """Returns a dataframe containing the number of each type in each record.

        Parameters
        ----------
        rec_ids_list : list, optional
            List of record ids, by default DS1
        cols : list
            List of labels classes, by default None

        Returns
        -------
        pandas.dataframe
            Pandas dataframe of count of each label type.
        """

        if cols == None:
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

    def stndr(self, arr, mean, std):
        X = arr.copy()
        X = (X - np.mean(X)) / np.std(X)
        return X

    def slice_data(self, ds, labels_list):
        # only keep the lables in lable_list
        # ds = copy.copy(ds)
        sliced_x = ds["waveforms"]
        sliced_r = ds["beat_feats"]
        sliced_y = ds["labels"]
        indexes_keep = []
        for sym in labels_list:
            inds = [i for i, item in enumerate(sliced_y) if item == sym]
            indexes_keep.extend(inds)
        sliced_x = sliced_x[indexes_keep]
        sliced_r = sliced_r[indexes_keep]
        sliced_y = sliced_y[indexes_keep]
        # sliced_y = [sliced_y[i] for i in indexes_keep]

        return {"waveforms": sliced_x, "beat_feats": sliced_r, "labels": sliced_y}

    def binarize_lables(self, y, positive_lable, pos=1, neg=-1):
        # y a list of lables
        # positive_lable: positive class lable
        new_y = [pos if item == positive_lable else neg for item in y]
        return new_y