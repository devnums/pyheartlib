import os
import yaml
from tqdm import tqdm
from abc import ABC, abstractmethod
from pyheartlib.io import get_data, save_data
from pyheartlib.processing import Processing


class Data:
    """Parent of other data classes.

    Parameters
    ----------
    base_path : str, optional
        Path of the main directory for loading and saving data, by default None
    remove_bl : bool, optional
        If True, the baseline is removed from the raw signals before extracting
        beat excerpts, by default False
    lowpass : bool, optional
        Whether to apply low pass filtering to the raw signals, by default False
    sampling_rate : int, optional
        Sampling rate of the signals, by default 360
    cutoff : int, optional
        Parameter of the low pass filter, by default 45
    order : int, optional
        Parameter of the low pass filter, by default 15

    Attributes
    ----------
    config : dict
        Dataset config loaded from config.yaml file.
    data_path : str
        Path of the data directory.

    """

    def __init__(
        self,
        base_path=None,
        remove_bl=False,
        lowpass=False,
        sampling_rate=360,
        cutoff=45,
        order=15,
    ):
        self.remove_bl = remove_bl
        self.lowpass = lowpass
        self.sampling_rate = sampling_rate
        self.cutoff = cutoff
        self.order = order

        if base_path is None:
            self.base_path = os.getcwd()
            conf_path = os.path.join(self.base_path, "src", "pyheartlib", "config.yaml")
        else:
            self.base_path = base_path
            conf_path = os.path.join(self.base_path, "config.yaml")
        with open(conf_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.data_path = os.path.join(self.base_path, self.config["DATA_DIR"])

    def get_ecg_record(self, record_id=106):
        """Loads a record and returns its components.

        Parameters
        ----------
        record_id : str
            Record id.
        return_dict : bool
            If True the method returns a dictionary, otherwise returns a pandas dataframe.

        Returns
        -------
        dict or dataframe
            If return_dict is True, it returns a dictionary
            with keys: *signal*, *r_locations*, *r_labels*, *rhythms*,
            *rhythms_locations*.

            If return_dict is False, it returns a dataframe containing the time, raw signal, and
            a list of equal size to the raw signal with None values except at annotations locations.
        """
        record_path = os.path.join(self.data_path, str(record_id))
        data_dict = get_data(record_path, self.config, return_dict=True)
        data_dict["signal"] = Processing.denoise_signal(
            data_dict["signal"],
            remove_bl=self.remove_bl,
            lowpass=self.lowpass,
            sampling_rate=self.sampling_rate,
            cutoff=self.cutoff,
            order=self.order,
        )
        return data_dict


class DataSeq(ABC):
    """Parent of other data classes."""

    progress_bar = True

    @abstractmethod
    def full_annotate(self):
        pass

    def get_all_annotated_records(self, rec_list):
        """Creates full annotation for records in the provided list.

        Parameters
        ----------
        rec_list : list
            List of records.

        Returns
        -------
        list
            A list containing a dict for each record. [rec1,rec2,....].
            Each rec is a dictionary with keys: *signal*, *r_locations*,
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
    def make_samples_info(self):
        pass

    def save_samples(self, rec_list, file_name, win_size, stride):
        """Returns and saves the signals and their full annotations along
        with information necessary for extracting signal excerpts.

        Parameters
        ----------
        rec_list : list
            Contains ids of records
        file_name : str, optional
            Save file name, by default None
        win_size : int, optional
            Windows size
        stride : int, optional
            Stride of the moving windows, by default 36

        Returns
        -------
        list
            The list contains two elements. First element is a list containing
            a dict for each record, [rec1,rec2,....].
            Each rec is a dict with keys: *signal*, *r_locations*, *r_labels*,
            *rhythms*, *rhythms_locations*, *full_ann*. Second element is a nested list.
            Each inner list is like [record_no, start_win, end_win, label].
            E.g. : [[10,500,800,'AFIB'], [10,700,900,'(N'], ...].
        """

        annotated_records = self.get_all_annotated_records(rec_list)
        samples_info = self.make_samples_info(annotated_records, win_size, stride)
        data = [annotated_records, samples_info]
        file_path = os.path.join(self.base_path, file_name)
        save_data(data, file_path=file_path)
        return data
