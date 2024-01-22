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
from abc import ABC, abstractmethod
from warnings import warn

import numpy as np
import yaml
from tqdm import tqdm

from pyheartlib.io import get_data, save_data
from pyheartlib.processing import Processing


class Data:
    """Parent of other data classes.

    Parameters
    ----------
    base_path : str, optional
        Path of the main directory for storing the original and
        processed data, by default None
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
    processors : list, optional
        Ordered list of functions' names for preprocessing the raw signals.
        Each function takes a one-dimensional NumPy array as its input and
        returns an array of the same length.

    Attributes
    ----------
    config : dict
        Dataset config. It is loaded from the config.yaml file.
    data_path : str
        Path of the data directory.
    sampling_rate : int
        Sampling rate of the signals, by default 360

    """

    config = dict()
    data_path = ""
    sampling_rate = 360

    def __init__(
        self,
        base_path=None,
        remove_bl=False,
        lowpass=False,
        cutoff=45,
        order=15,
        **kwargs,
    ):
        self.remove_bl = remove_bl
        self.lowpass = lowpass
        self.cutoff = cutoff
        self.order = order
        self.kwargs = kwargs

        if base_path is None:
            self.base_path = os.getcwd()
            conf_path = os.path.join(
                self.base_path, "src", "pyheartlib", "config.yaml"
            )
        else:
            self.base_path = base_path
            conf_path = os.path.join(self.base_path, "config.yaml")
        with open(conf_path) as f:
            __class__.config = yaml.load(f, Loader=yaml.FullLoader)
        __class__.data_path = os.path.join(
            self.base_path, self.config["DATA_DIR"]
        )
        __class__.sampling_rate = int(self.config["SAMPLING_RATE"])

    def get_ecg_record(self, record_id=106):
        """Returns a record as a dictionary.

        Parameters
        ----------
        record_id : str
            Record id.

        Returns
        -------
        dict
            A dictionary with keys: *signal*, *r_locations*, *r_labels*,
            *rhythms*, *rhythms_locations*.

        """
        record_path = os.path.join(self.data_path, str(record_id))
        data_dict = get_data(record_path, self.config, return_dict=True)
        signal = data_dict["signal"]
        processed_sig = np.empty_like(signal)
        for ch in range(signal.shape[1]):
            processed_sig[:, ch] = Processing.denoise_signal(
                signal[:, ch],
                remove_bl=self.remove_bl,
                lowpass=self.lowpass,
                sampling_rate=self.sampling_rate,
                cutoff=self.cutoff,
                order=self.order,
            )
            if "processors" in self.kwargs.keys():
                try:
                    processed_sig[:, ch] = Processing.custom_processors(
                        signal[:, ch], processors=self.kwargs["processors"]
                    )
                except Exception:
                    errmsg = (
                        "Processors error. Processors parameter"
                        " must be a list of function names."
                    )
                    raise Exception(errmsg)

        data_dict["signal"] = processed_sig
        return data_dict


class DataSeq(ABC):
    """Parent of other data classes."""

    progress_bar = True

    @abstractmethod
    def full_annotate(self):
        pass

    def get_all_records(self, rec_list):
        """Returns all the records.

        Parameters
        ----------
        rec_list : list
            List of records IDs.

        Returns
        -------
        list
            A list containing a dictionary for each record. [rec1,rec2,....].
            Each record is a dictionary with keys: *signal*, *r_locations*,
            *r_labels*, *rhythms*, *rhythms_locations*, *full_ann*.

        """

        all_recs = []
        for rec_id in tqdm(rec_list, disable=DataSeq.progress_bar):
            rec_dict = self.get_ecg_record(record_id=rec_id)
            rec_dict["full_ann"] = self.full_annotate(rec_dict)[
                1
            ]  # adding this list to the dict
            all_recs.append(rec_dict)
        return all_recs

    @abstractmethod
    def gen_samples_info(self):
        pass

    def save_dataset(self, rec_list, file_name, win_size, stride, **kwargs):
        """Saves a dataset holding ECG records along with metadata
        about signal excerpts.

        Parameters
        ----------
        rec_list : list
            List of records IDs.
        file_name : str, optional
            Name of the file that will be saved, by default None
        win_size : int, optional
            Sliding window length (excerpt length).
        stride : int, optional
            Stride of the sliding window, by default 36

        Returns
        -------
        tuple
            Two items: (records, metadata)

            The first item is a list of records ([rec1_dict, ...]).
            Each record is a dictionary with keys: *signal*, *r_locations*,
            *r_labels*, *rhythms*, *rhythms_locations*, *full_ann*.

            The second item is a nested list. Each inner list contains the
            metadata for an excerpt structured as:
            [record_id, onset, offset, annotation].
            E.g. : [[10,500,800,'AFIB'], ...].
        """

        kargs = {"return_ds": False}  # default values
        kargs.update(kwargs)
        return_ds = kargs["return_ds"]

        annotated_records = self.get_all_records(rec_list)
        samples_info = self.gen_samples_info(
            annotated_records, win_size, stride, **kwargs
        )
        data = (annotated_records, samples_info)
        file_path = os.path.join(self.base_path, file_name)
        save_data(data, file_path=file_path)
        if return_ds:
            return data
        else:
            return None

    def save_samples(self, *args, **kwargs):
        msg = "Method save_samples() will be deprecated.Use save_dataset()."
        warn(msg, DeprecationWarning, stacklevel=2)
        return self.save_dataset(*args, **kwargs)
