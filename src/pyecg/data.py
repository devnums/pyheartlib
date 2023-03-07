import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyecg.dataset_config import DATA_DIR
from pyecg.io import get_data, save_data, load_data
from pyecg.processing import Processing


class Data:
    """Parent of other data classes."""

    def __init__(
        self,
        base_path=os.getcwd(),  # TODO look or create data dir
        data_path=DATA_DIR,
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

        self.base_path = base_path
        self.data_path = os.path.join(self.base_path, data_path)
        self.remove_bl = remove_bl
        self.lowpass = lowpass
        self.sampling_rate = sampling_rate
        self.cutoff = cutoff
        self.order = order

    def get_ecg_record(self, record_id=106):
        """Loads a record and return its components.

        Parameters
        ----------
        record_id : str
            Record id.
        return_dict : bool
            If True returns as a dict otherwise returns as a pandas dataframe.

        Returns
        -------
        dict or dataframe
            If return_dict is True, it returns a dictionary
            with keys: 'signal','r_locations','r_labels','rhythms','rhythms_locations'.
            If return_dict is False, it returns a dataframe containing the time, raw signal, and
            a list of equal size to the raw signal with None values except at anntations locations.
        """
        record_path = os.path.join(self.data_path, str(record_id))
        data_dict = get_data(record_path, return_dict=True)
        data_dict["signal"] = Processing.denoise_signal(
            data_dict["signal"],
            remove_bl=self.remove_bl,
            lowpass=self.lowpass,
            sampling_rate=self.sampling_rate,
            cutoff=self.cutoff,
            order=self.order,
        )
        return data_dict
