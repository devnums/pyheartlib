import os
import math
import time
import numpy as np
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from pyecg.data import Data, DataSeq
from pyecg.dataset_config import DATA_DIR, DS1
from pyecg.io import get_data, save_data
from pyecg.features import get_hrv_features


class ArrhythmiaData(Data,DataSeq):
    """Provides data for Arrhythmia classficaation."""

    def __init__(
        self,
        base_path=os.getcwd(),
        data_path=DATA_DIR,
        remove_bl=False,
        lowpass=False,
        sampling_rate=360,
        cutoff=45,
        order=15,
    ):
        super().__init__(
            base_path,
            data_path,
            remove_bl,
            lowpass,
            sampling_rate,
            cutoff,
            order,
        )

    def full_annotate(self, record):
        """Fully annotate a signal.

        Parameters
        ----------
        record : dict
            Record as a dictionary with keys: 'signal','r_locations','r_labels','rhythms','rhythms_locations'.

        Returns
        -------
        list
            A list of signal and full_ann: [signal,full_ann]. First element is the original signal(1D ndarray).
            Second element is a list that has the same size as the signal with
            arrhythmia types at each index: ['(N','(N','(N','(N','AFIB','AFIB','AFIB',...].
        """

        signal, _, _, rhythms, rhythms_locations = record.values()
        sig_length = len(signal)
        full_ann = []
        full_ann = ["unlab"] * len(signal)
        for i in range(len(rhythms_locations)):
            remained = sig_length - rhythms_locations[i]
            full_ann[rhythms_locations[i] :] = [rhythms[i]] * remained
        record_full = [signal, full_ann]
        return record_full

    def make_samples_info(self, annotated_records, win_size=30 * 360, stride=36):
        """Creates a list of signal excerpts and their labels. For each excerpt the
            record id, start point, and end point of it on the original signal is extracted.

        Parameters
        ----------
        annotated_records : list
            A list containing a dict for each record. [rec1,rec2,....].
            Each rec is a dict with keys: 'signal','r_locations','r_labels','rhythms','rhythms_locations', 'full_ann'.
        win_size : int, optional
            Windows size, by default 30*360
        stride : int, optional
            Stride, by default 36

        Returns
        -------
        list
            A list of lists. Each inner list is like [record_no, start_win, end_win, label].
            E.g. : [[10,500,800,'AFIB'], [10,700,900,'(N'], ...]
        """

        stride = int(stride)
        win_size = int(win_size)

        samples_info = []
        for rec_no in tqdm(range(len(annotated_records))):
            signal = annotated_records[rec_no]["signal"]
            full_ann = annotated_records[rec_no]["full_ann"]
            assert len(signal) == len(
                full_ann
            ), "signal and annotation must have the same length!"

            end = win_size
            while end < len(full_ann):
                start = int(end - win_size)
                # unique arrhythmia type in each segment
                if len(set(full_ann[start:end])) == 1:
                    label = full_ann[start]
                    samples_info.append([rec_no, start, end, label])
                end += stride
            time.sleep(3)
        return samples_info


class ECGSequence(Sequence):
    def __init__(
        self,
        data,
        samples_info,
        class_labels=None,
        batch_size=128,
        shuffle=True,
        denoise=True,
    ):
        """
        Parameters
        ----------
        data : list
                        A list containing a dict for each record, [rec1,rec2,....].
                        Each rec is a dict with keys: 'signal','r_locations','r_labels','rhythms','rhythms_locations', 'full_ann'.
        samples_info : list
                        A list of lists. Each inner list is like [record_no, start_win, end_win, label].
                        E.g. : [[10,500,800,'AFIB'], [10,700,900,'(N'], ...].
        class_labels : list, optional
                        List of arrhythmia classes in the data, by default None
        batch_size : int, optional
                        Batch size, by default 128
        shuffle : bool, optional
                        If True shuffle the sample data, by default True
        denoise : bool, optional
                        If True denoise the signals, by default True
        """
        self.shuffle = shuffle
        self.denoise = denoise
        self.batch_size = batch_size
        self.data = data
        self.samples_info = samples_info
        self.class_labels = class_labels
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.samples_info) / self.batch_size)

    def __getitem__(self, idx):
        batch_samples = self.samples_info[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        batch_seq = []
        batch_label = []
        batch_rri = []
        for sample in batch_samples:
            # eg sample:[10,500,800,'AFIB'] ::: [rec,start,end,label]
            rec_no = sample[0]
            start = sample[1]
            end = sample[2]
            label = sample[3]
            if self.class_labels != None:
                label = self.get_integer(label)

            seq = self.data[rec_no]["signal"][start:end]

            batch_seq.append(seq)
            batch_label.append(label)

            rri = self.get_rri(rec_no, start, end)
            batch_rri.append(rri)

        batch_rri_feat = self.get_rri_features(np.array(batch_rri) * 1000)

        # return np.array(batch_seq),np.array(batch_label)
        return [np.array(batch_seq), np.array(batch_rri), batch_rri_feat], np.array(
            batch_label
        )

    def on_epoch_end(self):
        """After each epoch shuffles the samples."""
        if self.shuffle:
            np.random.shuffle(self.samples_info)

    def get_integer(self, label):
        """Converts text label to integer.

        Parameters
        ----------
        label : str
                String label.

        Returns
        -------
        int
                Integer label corresponding to the str label.
        """
        return self.class_labels.index(label)

    def get_rri(self, rec_no, start, end):
        """Computes RR intervals.
        TOdo
        Parameters
        ----------
        rec_no : _type_
                _description_
        start : _type_
                _description_
        end : _type_
                _description_

        Returns
        -------
        _type_
                _description_
        """
        r_locations = np.asarray(self.data[rec_no]["r_locations"])  # entire record
        inds = np.where((r_locations >= start) & (r_locations < end))
        rpeak_locs = list(r_locations[inds])
        rri = [
            (rpeak_locs[i + 1] - rpeak_locs[i]) / 360.0
            for i in range(0, len(rpeak_locs) - 1)
        ]
        # padding for 30sec---len=150
        # print(rri)
        rri_zeropadded = np.zeros(150)
        rri_zeropadded[: len(rri)] = rri
        # print(rri_zeropadded)
        rri_zeropadded = rri_zeropadded.tolist()
        rri_zeropadded = rri_zeropadded[:20]  # TODO

        return rri_zeropadded

    def get_rri_features(self, arr):
        """_summary_

        Parameters
        ----------
        arr : _type_
                _description_

        Returns
        -------
        _type_
                _description_
        """
        # features = ['max','min']
        return get_hrv_features(arr)
